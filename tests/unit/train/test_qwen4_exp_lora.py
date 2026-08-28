import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import (
    LoRAState,
    _find_target_modules,
    _target_patterns,
    save_lora_config,
)
from prime_rl.trainer.models.layers.lora import MultiLoRALinear, set_lora_num_tokens
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAFusedGateUpGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.qwen4_exp import Qwen4ExpPreTrainedModel


class _Qwen4ExpTargets(nn.Module):
    default_lora_target_modules = (
        r"(?:^|\.)self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        r"(?:^|\.)linear_attn\.(in_proj_z|in_proj_b|in_proj_a|out_proj)$",
        r"(?:^|\.)ple\.(key_proj|value_proj)$",
        r"(?:^|\.)shared_expert\.(w1|w2|w3)$",
        r"(?:^|\.)mlp\.experts$",
    )
    lora_grouped_experts_cls = MultiLoRAFusedGateUpGroupedExperts

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="Qwen/Qwen3.8-Flash-Next")
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(8, 8, bias=False),
                "k_proj": nn.Linear(8, 8, bias=False),
                "v_proj": nn.Linear(8, 8, bias=False),
                "o_proj": nn.Linear(8, 8, bias=False),
                "indexer": nn.ModuleDict({"index_qk_proj": nn.Linear(8, 8, bias=False)}),
            }
        )
        self.linear_attn = nn.ModuleDict(
            {
                "in_proj_qkv": nn.Linear(8, 24, bias=False),
                "in_proj_z": nn.Linear(8, 8, bias=False),
                "in_proj_b": nn.Linear(8, 2, bias=False),
                "in_proj_a": nn.Linear(8, 2, bias=False),
                "out_proj": nn.Linear(8, 8, bias=False),
            }
        )
        self.ple = nn.ModuleDict({"key_proj": nn.Linear(8, 8, bias=False), "value_proj": nn.Linear(8, 8, bias=False)})
        self.shared_expert = nn.ModuleDict(
            {
                "w1": nn.Linear(8, 12, bias=False),
                "w2": nn.Linear(12, 8, bias=False),
                "w3": nn.Linear(8, 12, bias=False),
            }
        )
        self.mlp = nn.ModuleDict({"experts": GroupedExperts(8, 12, 2, use_grouped_mm=False)})

    @staticmethod
    def lora_adapter_target_module(name: str) -> str:
        projections = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        target = name.rsplit(".", 1)[-1]
        return projections[target] if name.startswith("shared_expert.") else target


def test_qwen4_exp_default_lora_targets_are_architecture_aware():
    model = _Qwen4ExpTargets()

    targets = _find_target_modules(model, _target_patterns(model, LoRAConfig()))

    assert "self_attn.indexer.index_qk_proj" not in targets
    assert "linear_attn.in_proj_qkv" not in targets
    assert "mlp.experts" in targets
    assert {"shared_expert.w1", "shared_expert.w2", "shared_expert.w3"} <= set(targets)
    assert {"ple.key_proj", "ple.value_proj"} <= set(targets)


def test_qwen4_exp_expert_lora_has_gradients_and_native_state(tmp_path):
    torch.manual_seed(0)
    config = LoRAConfig(rank=2, alpha=2)
    state = LoRAState(config, torch.device("cpu"))
    experts = GroupedExperts(8, 12, 2, use_grouped_mm=False)
    experts.init_weights(0.02)
    lora = MultiLoRAFusedGateUpGroupedExperts(experts, rank=2, n_adapters=1, alpha=2, use_grouped_mm=False)
    for name, parameter in lora.named_parameters_for_adapter(0):
        if name.endswith("lora_B"):
            nn.init.normal_(parameter)
    set_lora_num_tokens(torch.tensor([4], dtype=torch.int32))

    lora(torch.randn(4, 8), torch.tensor([2, 2])).square().mean().backward()

    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for _, parameter in lora.named_parameters_for_adapter(0)
    )

    state.register_module("model.layers.0.mlp.experts", lora)
    state.register_adapter_state_dict_converter(Qwen4ExpPreTrainedModel.convert_adapter_to_hf)
    adapter = state.adapter_state_dict()
    expected = {
        "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight",
        "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
    }
    assert set(adapter) == expected
    assert adapter[next(key for key in adapter if "base_layer.lora_A" in key)].shape == (4, 8)
    assert adapter[next(key for key in adapter if "base_layer.lora_B" in key)].shape == (24, 4)

    path = tmp_path / "adapter_model.safetensors"
    save_file(adapter, path)
    restored = load_file(path)
    assert set(restored) == expected
    for key in expected:
        torch.testing.assert_close(restored[key], adapter[key])


def test_qwen4_exp_adapter_config_uses_official_expert_parameters(tmp_path):
    model = _Qwen4ExpTargets()
    config = LoRAConfig(rank=2, alpha=2)
    LoRAState(config, torch.device("cpu"))
    model.mlp["experts"] = MultiLoRAFusedGateUpGroupedExperts(
        model.mlp["experts"], rank=2, n_adapters=1, alpha=2, use_grouped_mm=False
    )
    for name in ("w1", "w2", "w3"):
        model.shared_expert[name] = MultiLoRALinear(
            model.shared_expert[name], rank=2, n_adapters=1, alpha=2, use_grouped_mm=False
        )

    save_lora_config(model, tmp_path, rank=2, alpha=2, dropout=0)

    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["target_modules"] == ["down_proj", "gate_proj", "up_proj"]
    assert saved["target_parameters"] == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]


def test_qwen4_exp_adapter_converter_renames_shared_experts():
    state = {
        "model.layers.3.shared_expert.w1.lora_A.weight": torch.randn(2, 8),
        "model.layers.3.shared_expert.w2.lora_B.weight": torch.randn(8, 2),
        "model.layers.3.shared_expert.w3.lora_A.weight": torch.randn(2, 8),
    }

    converted = Qwen4ExpPreTrainedModel.convert_adapter_to_hf(state)

    assert set(converted) == {
        "base_model.model.model.layers.3.mlp.shared_expert.gate_proj.lora_A.weight",
        "base_model.model.model.layers.3.mlp.shared_expert.down_proj.lora_B.weight",
        "base_model.model.model.layers.3.mlp.shared_expert.up_proj.lora_A.weight",
    }

import importlib
import sys
from types import SimpleNamespace

import torch

from prime_rl.trainer.models.qwen4_exp import Qwen4ExpConfig, Qwen4ExpForCausalLM


class FakeMuon:
    def __init__(self, *, params, **kwargs):
        self.param_groups = params


class FakeParallelDims:
    ep_enabled = True
    dp_shard_enabled = False
    cp_enabled = False
    dp_replicate_enabled = False
    world_mesh = "world-mesh"

    def get_mesh(self, name: str):
        return name


def _model():
    config = Qwen4ExpConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention", "linear_attention"],
        ple_layer_ids=[2],
        ple_embed_dim=16,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=2,
        use_grouped_mm=False,
    )
    return Qwen4ExpForCausalLM(config)


def test_qwen4exp_muon_groups_shard_incompatible_matrices_with_adamw(monkeypatch):
    monkeypatch.setitem(sys.modules, "dion", SimpleNamespace(Muon=FakeMuon))
    sys.modules.pop("prime_rl.trainer.optim", None)
    optim = importlib.import_module("prime_rl.trainer.optim")
    model = _model().to(torch.bfloat16)
    for layer in model.model.layers:
        layer.mlp.router.to(torch.float32)
    named_params = list(model.named_parameters())
    config = SimpleNamespace(type="muon", lr=1e-3, weight_decay=0.1, mu=0.95, betas1=0.9, betas2=0.95)

    optimizer = optim._create_muon_optimizer(
        config,
        named_params,
        FakeParallelDims(),
        lr=config.lr,
        adamw_parameter_names=model.muon_adamw_parameter_names(),
    )
    names_by_id = {id(param): name for name, param in named_params}
    assignments = {
        names_by_id[id(param)]: (group["algorithm"], group.get("distributed_mesh_name"))
        for group in optimizer.param_groups
        for param in group["params"]
    }

    adamw_names = {name for name, (algorithm, _) in assignments.items() if algorithm == "adamw"}
    assert model.muon_adamw_parameter_names() | {"lm_head.weight"} <= adamw_names
    assert all(name in adamw_names for name in assignments if name.endswith("shared_expert_gate.weight"))
    gdn_gates = {n for n in assignments if n.endswith(("in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight"))}
    assert gdn_gates and gdn_gates <= adamw_names
    assert all(assignments[name][0] == "muon" for name in assignments if name.endswith("block_inject_weight.weight"))
    assert all(len({param.dtype for param in group["params"]}) == 1 for group in optimizer.param_groups)

    expert_names = {name for name in assignments if ".mlp.experts." in name}
    assert expert_names
    assert all(assignments[name] == ("muon", "dp_shard_mod_ep") for name in expert_names)
    assert assignments["model.layers.0.self_attn.q_proj.weight"][0] == "muon"
    assert assignments["model.layers.1.linear_attn.in_proj_qkv.q_proj.weight"][0] == "muon"

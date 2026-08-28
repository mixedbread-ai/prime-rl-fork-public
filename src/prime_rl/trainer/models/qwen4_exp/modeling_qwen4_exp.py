import functools
import re
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import init
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAFusedGateUpGroupedExperts
from prime_rl.trainer.models.layers.moe import FeedForward, GroupedExperts, MoE, MoEArgs
from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRotaryEmbedding,
    normalize_qwen3_5_attn_implementation,
)
from prime_rl.trainer.models.qwen3_5_moe.mrope import build_qwen3_5_mrope_position_ids
from prime_rl.utils.cp import setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

from .configuration_qwen4_exp import Qwen4ExpConfig
from .converting_qwen4_exp import conversion_chain
from .norms import Qwen4ExpRMSNorm
from .ple import Qwen4ExpNGramEmbedding, Qwen4ExpPLELayer
from .qsa_attention import QsaLayout, Qwen4ExpSparseAttention, build_qsa_layout


class SplitQKVProjection(nn.Module):
    def __init__(self, hidden_size: int, key_dim: int, value_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, value_dim, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch.cat((self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)), dim=-1)


class Qwen4ExpGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpConfig, use_inject: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_inject else None

    def forward(self, hyper_input: Tensor):
        hyper_input_normed = self.hc_norm(hyper_input)
        mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        mix_weight = torch.sigmoid(self.input_mix_weight_up(mix_weight)).unflatten(-1, (self.hc_count, -1))
        streams = hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (mix_weight * streams).mean(dim=-2)

        if self.block_inject_weight is None:
            return mixed_input

        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, injection_weights


def write_to_streams(hyper_input: Tensor, block_output: Tensor, injection_weights: Tensor) -> Tensor:
    return hyper_input + (block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)


class Qwen4ExpGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    def __init__(self, config: Qwen4ExpConfig):
        super().__init__(config)
        self.in_proj_qkv = SplitQKVProjection(self.hidden_size, self.key_dim, self.value_dim)
        self.norm = FusedRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon, activation=config.output_gate_type)


QWEN4EXP_ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(Qwen4ExpSparseAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(Qwen4ExpSparseAttention, flash_attn_version=3),
    "flash_attention_4": functools.partial(Qwen4ExpSparseAttention, flash_attn_version=4),
}


def _scatter_visual_embeddings(
    inputs_embeds: Tensor,
    input_ids: Tensor,
    token_id: int,
    visual_embeds: Tensor,
    modality: str,
) -> Tensor:
    mask = input_ids == token_id
    if int(mask.sum()) != visual_embeds.shape[0]:
        raise ValueError(
            f"{modality} features and placeholder tokens do not match: "
            f"{visual_embeds.shape[0]} features for {int(mask.sum())} tokens"
        )
    return inputs_embeds.masked_scatter(mask.unsqueeze(-1).expand_as(inputs_embeds), visual_embeds)


def _visual_feature_count(grid_thw: Tensor | None, spatial_merge_size: int) -> int:
    if grid_thw is None:
        return 0
    return int((grid_thw.prod(-1) // spatial_merge_size**2).sum().item())


def _get_sparse_attention(config: Qwen4ExpConfig) -> nn.Module:
    attn_impl = normalize_qwen3_5_attn_implementation(config._attn_implementation)
    config._attn_implementation = attn_impl
    if attn_impl not in QWEN4EXP_ATTN_IMPL2CLASS:
        raise ValueError(
            f"Qwen4-Exp attention does not support '{attn_impl}'. "
            f"Supported implementations: {list(QWEN4EXP_ATTN_IMPL2CLASS)}."
        )
    return QWEN4EXP_ATTN_IMPL2CLASS[attn_impl](config)


class Qwen4ExpDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(config)
        else:
            self.self_attn = _get_sparse_attention(config)

        moe_args = MoEArgs(
            num_experts=config.num_experts,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=config.norm_topk_prob,
            route_scale=1.0,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=config.use_grouped_mm,
            load_balance_coeff=config.load_balance_coeff,
            fp8=getattr(config, "fp8", False),
        )
        self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        self.shared_expert = FeedForward(dim=config.hidden_size, hidden_dim=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        ple_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = Qwen4ExpPLELayer(config, ple_index) if ple_index is not None else None
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
        routed_experts: Optional[torch.LongTensor] = None,
        cu_seqlens_are_pre_shard: bool = False,
        qsa_layout: QsaLayout | None = None,
        ple_inputs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, ple_inputs)

        block_input, injection_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            block_output = self.linear_attn(
                block_input, cu_seqlens=cu_seqlens, cu_seqlens_are_pre_shard=cu_seqlens_are_pre_shard
            )
        else:
            block_output, _ = self.self_attn(
                block_input,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                qsa_layout=qsa_layout,
            )
        hidden_states = write_to_streams(hidden_states, block_output, injection_weights)

        block_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        router_input = block_input.reshape(-1, block_input.shape[-1])
        routed_output = self.mlp(block_input, routed_experts=routed_experts)
        shared_output = torch.sigmoid(self.shared_expert_gate(router_input)) * self.shared_expert(router_input)
        block_output = routed_output + shared_output.view_as(block_input)
        hidden_states = write_to_streams(hidden_states, block_output, injection_weights)
        return hidden_states


class Qwen4ExpPreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Qwen4ExpConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen4ExpDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {"hidden_states": Qwen4ExpDecoderLayer}
    supports_streaming_conversion = True
    default_lora_target_modules = (
        r"(?:^|\.)self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        r"(?:^|\.)linear_attn\.(in_proj_z|in_proj_b|in_proj_a|out_proj)$",
        r"(?:^|\.)ple\.(key_proj|value_proj)$",
        r"(?:^|\.)shared_expert\.(w1|w2|w3)$",
        r"(?:^|\.)mlp\.experts$",
    )
    lora_grouped_experts_cls = MultiLoRAFusedGateUpGroupedExperts

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, Qwen4ExpGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log, torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(0.01, 16).log_()
            )
        elif isinstance(module, (Qwen4ExpRMSNorm, Qwen3_5MoeRMSNorm)):
            # These scale by (1 + weight), so a centred gain is zero rather than one.
            nn.init.zeros_(module.weight)
        elif isinstance(module, (GroupedExperts, FeedForward)):
            module.init_weights(std)
        elif isinstance(module, Qwen4ExpPLELayer):
            nn.init.zeros_(module.conv1d.weight)
        elif isinstance(module, Qwen4ExpNGramEmbedding):
            nn.init.normal_(module.ngram_embedding, mean=0.0, std=std)
            module.reset_buffers()

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))

    def muon_adamw_parameter_names(self) -> set[str]:
        names = set()
        for module_name, module in self.named_modules():
            prefix = f"{module_name}." if module_name else ""
            if isinstance(module, nn.Linear) and module_name.endswith(("router.gate", "shared_expert_gate")):
                names.add(f"{prefix}weight")
                if module.bias is not None:
                    names.add(f"{prefix}bias")
            if isinstance(module, nn.Linear) and module_name.endswith(("input_mix_weight_down", "input_mix_weight_up")):
                names.add(f"{prefix}weight")
            if isinstance(module, nn.Conv1d):
                names.add(f"{prefix}weight")
                if module.bias is not None:
                    names.add(f"{prefix}bias")
        return names

    @staticmethod
    def lora_adapter_target_module(name: str) -> str:
        target = name.rsplit(".", 1)[-1]
        if "shared_expert" not in name.split("."):
            return target
        return {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[target]

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        attn_impl = normalize_qwen3_5_attn_implementation(attn_implementation or "flash_attention_3")
        if attn_impl not in QWEN4EXP_ATTN_IMPL2CLASS:
            raise ValueError(
                f"Qwen4-Exp attention does not support '{attn_implementation}'. "
                f"Supported implementations: {list(QWEN4EXP_ATTN_IMPL2CLASS)}."
            )
        return attn_impl

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "mlp.experts.gate_up_proj" in name
            or "mlp.experts.1.up_proj" in name
            or "mlp.shared_expert.gate_proj" in name
            or "ngram_embedding.shard_" in name
            for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in name or "mlp.router.gate.weight" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        projections = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        converted = {}
        for name, tensor in state_dict.items():
            for prime_name, hf_name in projections.items():
                name = re.sub(
                    rf"(\.layers\.\d+)\.shared_expert\.{prime_name}(?=\.)",
                    rf"\1.mlp.shared_expert.{hf_name}",
                    name,
                )
            converted[f"base_model.model.{name}"] = tensor
        return converted


class Qwen4ExpModel(Qwen4ExpPreTrainedModel):
    def __init__(self, config: Qwen4ExpConfig):
        config._attn_implementation = normalize_qwen3_5_attn_implementation(config._attn_implementation)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hc_count = config.hc_count
        self.indexer_compress_ratio = config.indexer_compress_ratio
        self.dense_equivalent_seqlen = config.dense_equivalent_seqlen
        self.has_sparse_attention = "full_attention" in config.layer_types
        self.has_ple = bool(config.ple_layer_ids)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen4ExpDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_inject=False)
        self.rotary_emb = Qwen3_5MoeRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self._cp_group: dist.ProcessGroup | None = None
        self._cp_rank = 0
        self._cp_world_size = 1

        self.post_init()

    @property
    def norm(self) -> nn.Module:
        return self.hyper_connection_mixer

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.cp_group = cp_group
                layer.linear_attn.cp_rank = cp_rank
                layer.linear_attn.cp_world_size = cp_world_size
            else:
                layer.self_attn.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)
            if layer.ple is not None:
                layer.ple.set_context_parallel_attributes(cp_group, cp_rank)

    def _full_position_ids(self, position_ids: Tensor) -> Tensor:
        if self._cp_world_size == 1:
            return position_ids
        gathered = [torch.empty_like(position_ids) for _ in range(self._cp_world_size)]
        dist.all_gather(gathered, position_ids.contiguous(), group=self._cp_group)
        return torch.cat(gathered, dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        ple_input_ids: Optional[torch.LongTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if ple_input_ids is None:
            if self.has_ple and input_ids is None:
                raise ValueError("Qwen4-Exp per-layer embeddings need `ple_input_ids` when called with `inputs_embeds`")
            ple_input_ids = input_ids

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        position_embeddings = self.rotary_emb(inputs_embeds, self._full_position_ids(position_ids))

        total_tokens = inputs_embeds.shape[1] * self._cp_world_size
        needs_selection = self.has_sparse_attention and (
            self._cp_world_size > 1 or max_seqlen > self.dense_equivalent_seqlen
        )
        qsa_layout = (
            build_qsa_layout(cu_seqlens, total_tokens, self.indexer_compress_ratio, self._cp_rank, self._cp_world_size)
            if needs_selection
            else None
        )

        hidden_states = inputs_embeds.repeat(1, 1, self.hc_count)
        ple_inputs = {
            layer_idx: layer.ple.prepare(ple_input_ids, cu_seqlens)
            for layer_idx, layer in enumerate(self.layers)
            if layer.ple is not None
        }
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_ple_inputs = ple_inputs.get(layer_idx)
            if layer_ple_inputs is not None:
                layer_ple_inputs = decoder_layer.ple.resolve(layer_ple_inputs)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                routed_experts=routed_experts[:, :, layer_idx, :] if routed_experts is not None else None,
                cu_seqlens_are_pre_shard=seq_lens_are_pre_shard,
                qsa_layout=qsa_layout,
                ple_inputs=layer_ple_inputs,
            )

        return MoeModelOutputWithPast(last_hidden_state=self.hyper_connection_mixer(hidden_states))


def _build_text_config(composite_config: PretrainedConfig) -> Qwen4ExpConfig:
    text_config = Qwen4ExpConfig(**composite_config.text_config.to_dict())
    attn_impl = getattr(
        composite_config.text_config, "_attn_implementation", getattr(composite_config, "_attn_implementation", None)
    )
    if attn_impl is not None:
        text_config._attn_implementation = attn_impl
    return text_config


class Qwen4ExpVLMModel(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Qwen4ExpModel(_build_text_config(config))

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.language_model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def _dummy_vision_inputs(self, device: torch.device) -> tuple[Tensor, Tensor]:
        vision_config = self.config.vision_config
        merge_size = vision_config.spatial_merge_size
        patch_dim = (
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * vision_config.patch_size
            * vision_config.patch_size
        )
        pixel_values = torch.zeros(merge_size * merge_size, patch_dim, device=device, dtype=self.visual.dtype)
        grid_thw = torch.tensor([[1, merge_size, merge_size]], dtype=torch.long, device=device)
        return pixel_values, grid_thw

    def prepare_inputs_embeds_and_position_ids(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        has_images = pixel_values is not None
        if has_images != (image_grid_thw is not None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if has_images and mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids is required for multimodal Qwen4-Exp inputs")
        if has_images:
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_grid_thw = image_grid_thw
        else:
            pixel_values, vision_grid_thw = self._dummy_vision_inputs(inputs_embeds.device)

        vision_output = self.visual(pixel_values, grid_thw=vision_grid_thw, return_dict=True)
        visual_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

        image_tokens = _visual_feature_count(image_grid_thw, self.visual.spatial_merge_size)
        if has_images:
            if visual_embeds.shape[0] != image_tokens:
                raise ValueError(
                    f"vision encoder returned {visual_embeds.shape[0]} features for {image_tokens} grid positions"
                )
            inputs_embeds = _scatter_visual_embeddings(
                inputs_embeds,
                input_ids,
                self.config.image_token_id,
                visual_embeds,
                "image",
            )
        else:
            inputs_embeds = inputs_embeds + visual_embeds.sum() * 0.0

        if position_ids is None:
            if image_grid_thw is not None:
                position_ids = build_qwen3_5_mrope_position_ids(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    seq_lens=seq_lens,
                )
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        return inputs_embeds, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> MoeModelOutputWithPast:
        inputs_embeds, position_ids = self.prepare_inputs_embeds_and_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            seq_lens=seq_lens,
        )
        ple_input_ids = input_ids

        cp_group = getattr(self.language_model, "_cp_group", None)
        if image_grid_thw is not None and cp_group is not None:
            cp_rank = self.language_model._cp_rank
            cp_world_size = self.language_model._cp_world_size
            setup_cp_attention_params(position_ids, cp_group=cp_group, cp_style="ulysses", seq_lens=seq_lens)
            inputs_embeds = shard_for_cp(inputs_embeds, cp_rank=cp_rank, cp_world_size=cp_world_size)
            ple_input_ids = shard_for_cp(ple_input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
            position_ids = shard_position_ids_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
            if routed_experts is not None:
                routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_world_size)
            seq_lens_are_pre_shard = True

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
            ple_input_ids=ple_input_ids,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )


class Qwen4ExpForCausalLM(Qwen4ExpPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _checkpoint_conversion_mapping = {}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._is_vlm = hasattr(config, "vision_config")
        self.supports_packed_multimodal_training = self._is_vlm

        if self._is_vlm:
            self.model = Qwen4ExpVLMModel(config)
            text_config = config.text_config
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
        else:
            self.model = Qwen4ExpModel(config)
            text_config = config

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def _language_model(self) -> Qwen4ExpModel:
        return self.model.language_model if self._is_vlm else self.model

    def _ple_prefixes(self) -> list[tuple[Qwen4ExpNGramEmbedding, str]]:
        model_prefix = "model.language_model" if self._is_vlm else "model"
        prefixes = []
        for layer_idx, layer in enumerate(self._language_model().layers):
            if layer.ple is not None:
                prefixes.append(
                    (
                        layer.ple.ple_embedding,
                        f"{model_prefix}.layers.{layer_idx}.ple.ple_embedding.ngram_embedding",
                    )
                )
        return prefixes

    def externalize_weights(self) -> None:
        for embedding, _ in self._ple_prefixes():
            embedding.externalize()

    def external_weight_prefixes(self) -> tuple[str, ...]:
        return tuple(f"{prefix}.shard_" for _, prefix in self._ple_prefixes())

    def bind_external_weights(self, snapshot_path: Path | None, mesh) -> None:
        group = mesh.get_group() if mesh is not None else None
        for embedding, prefix in self._ple_prefixes():
            embedding.bind_external_table(snapshot_path, prefix, group)

    def external_weight_state_dict(self) -> dict[str, Tensor]:
        state_dict = {}
        for embedding, prefix in self._ple_prefixes():
            if embedding.external_table is None:
                raise RuntimeError("Qwen4-Exp PLE table was not externalized")
            state_dict.update(
                {f"{prefix}.shard_{shard}.weight": tensor for shard, tensor in embedding.external_table.shards.items()}
            )
        return state_dict

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def _run_backbone(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        position_ids: Optional[torch.LongTensor],
        routed_experts: Optional[torch.LongTensor],
        pixel_values: Optional[Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        mm_token_type_ids: Optional[torch.LongTensor],
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool,
    ) -> MoeModelOutputWithPast:
        if not self._is_vlm:
            if any(x is not None for x in (pixel_values, image_grid_thw, mm_token_type_ids)):
                raise ValueError("this qwen4_exp config is text-only; it has no vision tower")
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
                seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            )

        if inputs_embeds is not None:
            raise ValueError("qwen4_exp builds its own multimodal input embeddings; pass input_ids instead")
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Union[torch.Tensor, None] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None, "use_cache is not supported for custom qwen4_exp for now"
        assert past_key_values is None, "past_key_values is not supported for custom qwen4_exp for now"

        outputs = self._run_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        language_model = self.model.language_model if self._is_vlm else self.model

        rope = language_model.rotary_emb
        inv_freq, rope.attention_scaling = rope.rope_init_fn(rope.config, rope.inv_freq.device)
        rope.inv_freq.copy_(inv_freq)

        for module in language_model.modules():
            if isinstance(module, Qwen4ExpNGramEmbedding):
                module.reset_buffers()

        if self._is_vlm:
            vision_rope = self.model.visual.rotary_pos_emb
            if hasattr(vision_rope, "inv_freq"):
                dim = vision_rope.inv_freq.shape[0]
                inv_freq = 1.0 / (
                    10000.0
                    ** (
                        torch.arange(0, dim * 2, 2, dtype=torch.float32, device=vision_rope.inv_freq.device) / (dim * 2)
                    )
                )
                vision_rope.inv_freq.copy_(inv_freq)


__all__ = [
    "Qwen4ExpForCausalLM",
    "Qwen4ExpModel",
    "Qwen4ExpPreTrainedModel",
]

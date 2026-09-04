import pytest
import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from prime_rl.trainer.models.qwen4_exp.converting_qwen4_exp import conversion_chain

SHARDS = 4
EXPERTS = 2
PLE_LAYER = 2


def _config() -> Qwen4ExpConfig:
    return Qwen4ExpConfig(
        hidden_size=8,
        num_hidden_layers=2,
        layer_types=["linear_attention", "linear_attention"],
        num_experts=EXPERTS,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=6,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        ple_layer_ids=[PLE_LAYER],
        split_ngram_parts=SHARDS,
        heads_per_ngram=1,
    )


def _prime_state(config: Qwen4ExpConfig, model_prefix: str) -> dict[str, torch.Tensor]:
    state = {"mtp.head.weight": torch.randn(4, 8)}
    for i in range(config.num_hidden_layers):
        p = f"{model_prefix}.layers.{i}"
        state |= {
            f"{p}.linear_attn.in_proj_qkv.q_proj.weight": torch.randn(8, 8),
            f"{p}.linear_attn.in_proj_qkv.k_proj.weight": torch.randn(8, 8),
            f"{p}.linear_attn.in_proj_qkv.v_proj.weight": torch.randn(8, 8),
            f"{p}.mlp.router.gate.weight": torch.randn(EXPERTS, 8),
            f"{p}.mlp.experts.w1": torch.randn(EXPERTS, 6, 8),
            f"{p}.mlp.experts.w2": torch.randn(EXPERTS, 8, 6),
            f"{p}.mlp.experts.w3": torch.randn(EXPERTS, 6, 8),
            f"{p}.shared_expert.w1.weight": torch.randn(6, 8),
            f"{p}.shared_expert.w2.weight": torch.randn(8, 6),
            f"{p}.shared_expert.w3.weight": torch.randn(6, 8),
            f"{p}.shared_expert_gate.weight": torch.randn(1, 8),
        }
        if i + 1 == PLE_LAYER:
            emb = f"{p}.ple.ple_embedding"
            state |= {
                f"{emb}.ngram_embedding": torch.randn(SHARDS, 5, 4),
                f"{emb}.layer_multipliers": torch.randint(1, 9, (1,)),
                f"{emb}.ngram_heads_vocab_sizes": torch.randint(1, 9, (2,)),
                f"{emb}.ngram_heads_offsets": torch.randint(1, 9, (2,)),
            }
    return state


@pytest.mark.parametrize("model_prefix", ["model", "model.language_model"])
def test_round_trip_recovers_every_key_the_model_needs(model_prefix):
    config = _config()
    state = _prime_state(config, model_prefix)
    chain = conversion_chain(config)

    recovered = apply_hf_to_prime(apply_prime_to_hf(dict(state), chain), chain)

    assert set(recovered) == set(state) - {"mtp.head.weight"}
    for key, value in recovered.items():
        assert torch.equal(value, state[key])

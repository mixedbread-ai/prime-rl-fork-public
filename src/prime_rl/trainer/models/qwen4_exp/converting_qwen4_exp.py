from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename, SplitConcat, Stack


def _conversion_chain(config, model_prefix: str) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"{model_prefix}.layers.{i}"
        if config.layer_types[i] == "linear_attention":
            ops.append(
                SplitConcat(
                    combined=f"{p}.linear_attn.in_proj_qkv.weight",
                    parts=[
                        (
                            f"{p}.linear_attn.in_proj_qkv.q_proj.weight",
                            config.linear_num_key_heads * config.linear_key_head_dim,
                        ),
                        (
                            f"{p}.linear_attn.in_proj_qkv.k_proj.weight",
                            config.linear_num_key_heads * config.linear_key_head_dim,
                        ),
                        (
                            f"{p}.linear_attn.in_proj_qkv.v_proj.weight",
                            config.linear_num_value_heads * config.linear_value_head_dim,
                        ),
                    ],
                    dim=0,
                )
            )
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(
            SplitConcat(
                combined=f"{p}.mlp.experts.gate_up_proj",
                parts=[(f"{p}.mlp.experts.w1", None), (f"{p}.mlp.experts.w3", None)],
                dim=1,
            )
        )
        ops.append(Rename(f"{p}.mlp.experts.down_proj", f"{p}.mlp.experts.w2"))
        ops.append(Rename(f"{p}.mlp.shared_expert.gate_proj.weight", f"{p}.shared_expert.w1.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert.down_proj.weight", f"{p}.shared_expert.w2.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert.up_proj.weight", f"{p}.shared_expert.w3.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert_gate.weight", f"{p}.shared_expert_gate.weight"))
        if i + 1 in config.ple_layer_ids:
            emb = f"{p}.ple.ple_embedding"
            ops.append(Stack(stacked=f"{emb}.ngram_embedding", item=emb + ".ngram_embedding.shard_{e}.weight"))
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    return [
        Drop("mtp.", is_prefix=True),
        *_conversion_chain(text_config, "model"),
        *_conversion_chain(text_config, "model.language_model"),
    ]

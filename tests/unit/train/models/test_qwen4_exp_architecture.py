import json
import threading
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
from prime_rl.trainer.models.qwen4_exp import qsa_attention
from prime_rl.trainer.models.qwen4_exp.configuration_qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpVisionConfig,
    Qwen4ExpVLMConfig,
)
from prime_rl.trainer.models.qwen4_exp.converting_qwen4_exp import conversion_chain
from prime_rl.trainer.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    SplitQKVProjection,
)
from prime_rl.trainer.models.qwen4_exp.ngram_table import ShardedNGramTable
from prime_rl.trainer.models.qwen4_exp.ple import Qwen4ExpNGramEmbedding
from prime_rl.trainer.models.qwen4_exp.qsa_attention import _rank_blocks
from prime_rl.utils.weights import convert_state_dict_streaming, load_state_dict


def _text_config(**kwargs):
    defaults = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention"] * 4,
        "ple_layer_ids": [1],
        "ple_embed_dim": 16,
        "ngram_vocab_size_base": 7,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 4,
        "heads_per_ngram": 2,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "eos_token_id": 2,
    }
    return Qwen4ExpConfig(**(defaults | kwargs))


def test_release_config_resolves_registered_composite(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text", "seed": None},
                "vision_config": {"model_type": "qwen4_exp", "out_hidden_size": 2560},
            }
        )
    )

    config = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(config, Qwen4ExpVLMConfig)
    assert isinstance(config.vision_config, Qwen4ExpVisionConfig)
    assert config.vision_config.model_type == "qwen4_exp_vision"
    assert config.vision_config.out_hidden_size == 2560
    assert config.text_config.seed == 1234
    assert config.to_dict()["vision_config"]["model_type"] == "qwen4_exp_vision"


def test_ple_table_is_frozen_and_hashes_to_the_expected_shape():
    embedding = Qwen4ExpNGramEmbedding(_text_config(), ple_layer_index=0)
    history = torch.tensor([[2, 2, 4, 5, 6]])
    reach = torch.tensor([[0, 1, 2]])

    output = embedding(history, reach)

    assert not embedding.ngram_embedding.requires_grad
    assert output.shape == (1, 3, 16)


def test_external_table_is_runtime_only_and_matches_contiguous_lookup(tmp_path):
    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = [torch.arange(i * 12, (i + 1) * 12, dtype=torch.float32).view(3, 4) for i in range(4)]
    save_file({f"{prefix}.shard_{i}.weight": shard for i, shard in enumerate(shards)}, tmp_path / "model.safetensors")
    table = ShardedNGramTable(
        shard_count=4, rows_per_shard=3, head_dim=4, dtype=torch.float32, init_std=0.02, seed=1234
    )
    table.bind(tmp_path, prefix)
    ids = torch.tensor([[0, 3, 11, 3]])

    output = table(ids)

    torch.testing.assert_close(output, torch.nn.functional.embedding(ids, torch.cat(shards)))

    embedding = Qwen4ExpNGramEmbedding(_text_config(), ple_layer_index=0)
    embedding.externalize()
    assert "ngram_embedding" not in embedding._parameters
    assert all("ngram_embedding" not in key for key in embedding.state_dict())
    embedding_source = tmp_path / "embedding"
    embedding_source.mkdir()
    save_file(
        {f"{prefix}.shard_{i}.weight": torch.zeros(12, 4) for i in range(4)},
        embedding_source / "model.safetensors",
    )
    embedding.bind_external_table(embedding_source, prefix)
    model = SimpleNamespace(_ple_prefixes=lambda: [(embedding, prefix)])
    external_state = Qwen4ExpForCausalLM.external_weight_state_dict(model)
    assert external_state.keys() == {f"{prefix}.shard_{i}.weight" for i in range(4)}
    assert all(not tensor.any() for tensor in external_state.values())


def test_external_table_lookup_is_prefetched_before_consumption(monkeypatch):
    table = ShardedNGramTable(
        shard_count=1, rows_per_shard=3, head_dim=2, dtype=torch.float32, init_std=0.02, seed=1234
    )
    table.shards = {0: torch.arange(6, dtype=torch.float32).view(3, 2)}
    started = threading.Event()
    release = threading.Event()
    lookup = table._lookup_cpu

    def delayed_lookup(ids, pin_memory=False):
        started.set()
        if not release.wait(5):
            raise TimeoutError("lookup was not released")
        return lookup(ids, pin_memory)

    monkeypatch.setattr(table, "_lookup_cpu", delayed_lookup)
    pending = table.start(torch.tensor([2, 0]))
    try:
        assert started.wait(1)
        assert not pending.done()
    finally:
        release.set()

    torch.testing.assert_close(pending.result(), torch.tensor([[4.0, 5.0], [0.0, 1.0]]))


def test_streaming_conversion_excludes_external_table(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prefix = "model.layers.0"
    save_file(
        {
            f"{prefix}.mlp.experts.gate_up_proj": torch.randn(2, 16, 16),
            **{f"{prefix}.ple.ple_embedding.ngram_embedding.shard_{i}.weight": torch.randn(3, 4) for i in range(4)},
        },
        source / "model-00001-of-00002.safetensors",
    )
    save_file(
        {f"{prefix}.mlp.experts.down_proj": torch.randn(2, 16, 8)},
        source / "model-00002-of-00002.safetensors",
    )
    output = tmp_path / "prime"
    config = _text_config(ple_layer_ids=[1])
    table_prefix = f"{prefix}.ple.ple_embedding.ngram_embedding.shard_"

    convert_state_dict_streaming(
        source,
        output,
        lambda state: apply_hf_to_prime(dict(state), conversion_chain(config)),
        exclude_prefixes=(table_prefix,),
    )
    converted = load_state_dict(output)

    assert f"{prefix}.mlp.experts.w1" in converted
    assert f"{prefix}.mlp.experts.w2" in converted
    assert f"{prefix}.mlp.experts.w3" in converted
    assert not any("ngram_embedding" in key for key in converted)
    manifest = output / "external_weights.json"
    assert manifest.exists()
    table = ShardedNGramTable(4, 3, 4, torch.float32, 0.02, 1234)
    table.bind(output, table_prefix.removesuffix(".shard_"))
    assert table(torch.tensor([0, 11])).shape == (2, 4)


def test_qsa_zero_score_ties_are_invariant_to_packed_prefixes():
    block_indices = torch.arange(513)
    base = _rank_blocks(torch.zeros(1, 513), torch.ones(1, 513, dtype=torch.bool), block_indices)
    base_selected = block_indices[base.topk(512, dim=-1).indices].sort().values

    prefixed_indices = torch.cat([torch.zeros(1, dtype=torch.long), block_indices])
    prefixed_allowed = torch.cat([torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, 513, dtype=torch.bool)], dim=1)
    prefixed = _rank_blocks(torch.zeros(1, 514), prefixed_allowed, prefixed_indices)
    prefixed_selected = prefixed_indices[prefixed.topk(512, dim=-1).indices].sort().values

    torch.testing.assert_close(prefixed_selected, base_selected)


def test_streaming_qsa_selection_matches_dense_ranking():
    torch.manual_seed(0)
    query = torch.randn(7, 3, 5)
    query[0].zero_()
    block_keys = torch.randn(21, 5)
    block_document = torch.repeat_interleave(torch.arange(3), torch.tensor([7, 8, 6]))
    block_index = torch.cat([torch.arange(7), torch.arange(8), torch.arange(6)])
    document_q = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    scored_blocks = torch.tensor([7, 4, 8, 2, 6, 3, 1])
    scaling = 5**-0.5
    topk = 4
    scores = qsa_attention._score_blocks(query, block_keys) * scaling
    allowed = (block_document == document_q[:, None]) & (block_index < scored_blocks[:, None])
    ranks = _rank_blocks(scores, allowed, block_index)
    values, indices = ranks.topk(topk, dim=-1)
    expected = block_index[indices]
    expected.masked_fill_(values == torch.iinfo(torch.int64).min, torch.iinfo(torch.int32).max)
    expected = expected.to(torch.int32).sort(dim=-1).values

    actual = qsa_attention._select_blocks(
        query,
        block_keys,
        document_q,
        scored_blocks,
        block_document,
        block_index,
        topk,
        scaling,
        query_chunk_size=2,
        key_chunk_size=3,
    )

    torch.testing.assert_close(actual, expected)


def test_streaming_qsa_selection_bounds_score_workspace(monkeypatch):
    query_chunk_size = 4
    key_chunk_size = 6
    score_shapes = []
    score_blocks = qsa_attention._score_blocks

    def record_score_shape(query, block_keys):
        score_shapes.append((query.shape[0], block_keys.shape[0]))
        return score_blocks(query, block_keys)

    monkeypatch.setattr(qsa_attention, "_score_blocks", record_score_shape)
    query = torch.randn(17, 2, 8)
    block_keys = torch.randn(29, 8)
    block_index = torch.arange(29)
    selected = qsa_attention._select_blocks(
        query,
        block_keys,
        torch.zeros(17, dtype=torch.long),
        torch.full((17,), 29),
        torch.zeros(29, dtype=torch.long),
        block_index,
        topk=5,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
    )

    assert selected.shape == (17, 5)
    assert max(rows for rows, _ in score_shapes) <= query_chunk_size
    assert max(columns for _, columns in score_shapes) <= key_chunk_size
    assert max(rows * columns for rows, columns in score_shapes) <= query_chunk_size * key_chunk_size


def test_compact_qsa_selection_matches_packed_cp_mask():
    layout = qsa_attention.build_qsa_layout(
        torch.tensor([0, 12, 28], dtype=torch.int32),
        total_kv=28,
        compress_ratio=4,
        cp_rank=1,
        cp_world_size=2,
    )
    selected = torch.tensor([[0, 2, torch.iinfo(torch.int32).max]] * 14, dtype=torch.int32)
    mask_mod = qsa_attention.QsaMaskMod()
    mask_mod.update(layout, selected)
    q_idx = torch.arange(14).repeat_interleave(28)
    kv_idx = torch.arange(28).repeat(14)
    block = layout.block_of_kv[kv_idx]
    expected = (
        (layout.document_q[q_idx] == layout.document_kv[kv_idx])
        & (q_idx + layout.query_offset >= kv_idx)
        & ((block >= layout.scored_blocks[q_idx]) | (selected[q_idx] == block[:, None]).any(dim=-1))
    )

    actual = mask_mod(torch.zeros_like(q_idx), torch.zeros_like(q_idx), q_idx, kv_idx)

    torch.testing.assert_close(actual, expected)
    assert not qsa_attention._contains_block(selected[:, :0], q_idx, block).any()


def test_qsa_selection_covers_padded_flex_attention_query_grid():
    config = _text_config(indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4)
    indexer = qsa_attention.Qwen4ExpQSAIndexer(config)
    layout = qsa_attention.build_qsa_layout(
        torch.tensor([0, 5], dtype=torch.int32),
        total_kv=5,
        compress_ratio=4,
        cp_rank=0,
        cp_world_size=1,
    )
    hidden_states = torch.randn(1, 5, config.hidden_size)
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    selection = indexer(hidden_states, position_embeddings, layout, None, 1)
    mask_mod = qsa_attention.QsaMaskMod()
    mask_mod.update(layout, selection.selected_blocks)

    assert selection.selected_blocks.shape[0] == qsa_attention.FLEX_BLOCK_SIZE
    assert not mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(127), torch.tensor(0))


def test_split_qkv_projection_trains_independent_maps():
    projection = SplitQKVProjection(hidden_size=4, key_dim=3, value_dim=5)
    hidden_states = torch.randn(2, 3, 4)

    output = projection(hidden_states)
    output.square().sum().backward()

    expected = torch.cat(
        (
            projection.q_proj(hidden_states),
            projection.k_proj(hidden_states),
            projection.v_proj(hidden_states),
        ),
        dim=-1,
    )
    torch.testing.assert_close(output, expected)
    assert all(parameter.grad is not None and parameter.grad.any() for parameter in projection.parameters())

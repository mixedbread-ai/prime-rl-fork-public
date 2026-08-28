from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from prime_rl.trainer.models.layers.attn import (
    flash_attn_3_varlen_func,
    flash_attn_4_varlen_func,
    flash_attn_varlen_func,
)
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedAttentionBase,
    apply_rotary_pos_emb,
)
from prime_rl.utils.cp import gather_for_cp, gather_for_cp_wo_grad

from .configuration_qwen4_exp import Qwen4ExpConfig
from .norms import Qwen4ExpRMSNorm

FLEX_BLOCK_SIZE = 128
INDEXER_QUERY_CHUNK = 2048
INDEXER_KEY_CHUNK = 2048
_UNSELECTABLE = torch.iinfo(torch.int64).min
_NO_BLOCK = torch.iinfo(torch.int32).max


def _rank_blocks(scores: Tensor, allowed: Tensor, block_index: Tensor) -> Tensor:
    key = (scores.view(torch.int32).to(torch.int64) << 32) - block_index
    return key.masked_fill(~allowed, _UNSELECTABLE)


def _score_blocks(query: Tensor, block_keys: Tensor) -> Tensor:
    return torch.relu(query.float() @ block_keys.float().T).sum(dim=1)


def _select_blocks(
    query: Tensor,
    block_keys: Tensor,
    document_q: Tensor,
    scored_blocks: Tensor,
    block_document: Tensor,
    block_index: Tensor,
    topk: int,
    scaling: float = 1.0,
    query_chunk_size: int = INDEXER_QUERY_CHUNK,
    key_chunk_size: int = INDEXER_KEY_CHUNK,
) -> Tensor:
    selected = block_index.new_full((query.shape[0], topk), -1, dtype=torch.int32)
    if not topk:
        return selected
    for query_start in range(0, query.shape[0], query_chunk_size):
        query_rows = slice(query_start, min(query_start + query_chunk_size, query.shape[0]))
        rows = query_rows.stop - query_rows.start
        best_ranks = block_index.new_full((rows, topk), _UNSELECTABLE)
        best_blocks = block_index.new_full((rows, topk), -1)
        for key_start in range(0, block_keys.shape[0], key_chunk_size):
            key_rows = slice(key_start, min(key_start + key_chunk_size, block_keys.shape[0]))
            scores = _score_blocks(query[query_rows], block_keys[key_rows]) * scaling
            allowed = (block_document[key_rows] == document_q[query_rows, None]) & (
                block_index[key_rows] < scored_blocks[query_rows, None]
            )
            ranks = _rank_blocks(scores, allowed, block_index[key_rows])
            blocks = block_index[key_rows].expand(rows, -1)
            candidates = torch.cat([best_ranks, ranks], dim=-1)
            candidate_blocks = torch.cat([best_blocks, blocks], dim=-1)
            best_ranks, indices = candidates.topk(topk, dim=-1)
            best_blocks = candidate_blocks.gather(1, indices)
        best_blocks.masked_fill_(best_ranks == _UNSELECTABLE, _NO_BLOCK)
        selected[query_rows] = best_blocks.to(torch.int32).sort(dim=-1).values
    return selected


def _contains_block(selected: Tensor, row: Tensor, block: Tensor) -> Tensor:
    if not selected.shape[-1]:
        return torch.zeros_like(block, dtype=torch.bool)
    low = torch.zeros_like(block)
    high = torch.full_like(block, selected.shape[-1])
    for _ in range(selected.shape[-1].bit_length()):
        middle = (low + high) // 2
        candidate = selected[row, middle.clamp(max=selected.shape[-1] - 1)]
        lower = candidate < block
        low = torch.where(lower, middle + 1, low)
        high = torch.where(lower, high, middle)
    return selected[row, low.clamp(max=selected.shape[-1] - 1)] == block


# Eager flex_attention ignores the block mask and materializes the whole score matrix.
_flex_attention = torch.compile(flex_attention)


def _apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rotary_dim = cos.shape[-1]
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat([rotated * cos + rotate_half(rotated) * sin, passthrough], dim=-1)


def _touched_kv_blocks(start: Tensor, span: int, kv_blocks: int) -> Tensor:
    width = -(-span // FLEX_BLOCK_SIZE) + 1
    first = start.unsqueeze(-1) // FLEX_BLOCK_SIZE
    last = (start + span - 1).unsqueeze(-1) // FLEX_BLOCK_SIZE
    return torch.minimum(first + torch.arange(width, device=start.device), last).clamp(max=kv_blocks - 1)


def _ordered_kv_blocks(present: Tensor) -> tuple[Tensor, Tensor]:
    dense = present.to(torch.int32)
    counts = dense.sum(dim=-1).to(torch.int32)
    order = torch.argsort(dense, dim=-1, descending=True, stable=True).to(torch.int32)
    return counts.view(1, 1, -1).contiguous(), order.view(1, 1, *order.shape).contiguous()


@dataclass(frozen=True)
class QsaLayout:
    document_q: Tensor
    document_kv: Tensor
    document_start_q: Tensor
    block_of_kv: Tensor
    scored_blocks: Tensor
    block_document: Tensor
    block_index: Tensor
    block_start: Tensor
    block_source: Tensor
    block_size: Tensor
    seq_lengths: tuple[int, int]
    query_offset: int
    q_blocks: int
    kv_blocks: int
    compress_ratio: int


def build_qsa_layout(
    cu_seqlens: Tensor, total_kv: int, compress_ratio: int, cp_rank: int, cp_world_size: int
) -> QsaLayout:
    device = cu_seqlens.device
    total_q = total_kv // cp_world_size
    query_offset = cp_rank * total_q

    positions = torch.arange(total_kv, device=device)
    boundaries = cu_seqlens.to(positions.dtype)
    document = torch.searchsorted(boundaries[1:], positions, right=True)
    local = positions - boundaries[document]
    block_of_kv = local // compress_ratio

    opens_block = local % compress_ratio == 0
    block_source = opens_block.cumsum(0) - 1
    block_index = block_of_kv[opens_block]
    block_size = torch.bincount(block_source, minlength=block_index.shape[0])
    query_slice = slice(query_offset, query_offset + total_q)
    document_q = document[query_slice]
    scored_blocks = (local[query_slice] + 1) // compress_ratio

    q_blocks = -(-total_q // FLEX_BLOCK_SIZE)
    kv_blocks = -(-total_kv // FLEX_BLOCK_SIZE)
    q_padding = q_blocks * FLEX_BLOCK_SIZE - total_q
    kv_padding = kv_blocks * FLEX_BLOCK_SIZE - total_kv

    return QsaLayout(
        document_q=F.pad(document_q, (0, q_padding), value=-1),
        document_kv=F.pad(document, (0, kv_padding), value=-2),
        document_start_q=F.pad(boundaries[document_q], (0, q_padding), value=0),
        block_of_kv=F.pad(block_of_kv, (0, kv_padding), value=0),
        scored_blocks=F.pad(scored_blocks, (0, q_padding), value=0),
        block_document=document[opens_block],
        block_index=block_index,
        block_start=positions[opens_block],
        block_source=block_source,
        block_size=block_size,
        seq_lengths=(total_q, total_kv),
        query_offset=query_offset,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        compress_ratio=compress_ratio,
    )


@dataclass(frozen=True)
class QsaSelection:
    selected_blocks: Tensor
    kv_num_blocks: Tensor
    kv_indices: Tensor


class QsaMaskMod:
    def __init__(self) -> None:
        self.layout: QsaLayout | None = None
        self.selected_blocks: Tensor | None = None

    def update(self, layout: QsaLayout, selected_blocks: Tensor) -> None:
        self.layout, self.selected_blocks = layout, selected_blocks

    def __call__(self, batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        layout = self.layout
        block = layout.block_of_kv[kv_idx]
        visible = (layout.document_q[q_idx] == layout.document_kv[kv_idx]) & (q_idx + layout.query_offset >= kv_idx)
        chosen = _contains_block(self.selected_blocks, q_idx, block)
        return visible & ((block >= layout.scored_blocks[q_idx]) | chosen)


class Qwen4ExpQSAIndexer(nn.Module):
    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.num_heads = config.indexer_n_heads
        self.num_key_value_heads = config.indexer_kv_heads
        self.head_dim = config.indexer_head_dim
        self.block_topk = config.block_topk
        self.scaling = self.head_dim**-0.5
        qk_dim = (self.num_heads + self.num_key_value_heads) * self.head_dim
        self.index_qk_proj = nn.Linear(config.hidden_size, qk_dim, bias=False)
        self.q_layernorm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.requires_grad_(False)

    def _pooled_block_keys(self, keys: Tensor, layout: QsaLayout, cos: Tensor, sin: Tensor) -> Tensor:
        pooled = keys.new_zeros(layout.block_size.shape[0], self.head_dim, dtype=torch.float32)
        pooled.index_add_(0, layout.block_source, keys.float())
        pooled = self.k_layernorm((pooled / layout.block_size.unsqueeze(-1)).to(keys.dtype))
        return _apply_partial_rope(
            pooled, cos.index_select(0, layout.block_start), sin.index_select(0, layout.block_start)
        )

    def _mark_reachable(self, present: Tensor, layout: QsaLayout, rows: slice, blocks: Tensor, kept: Tensor) -> None:
        positions = torch.arange(rows.start, rows.stop, device=blocks.device) + layout.query_offset
        document_start = layout.document_start_q[rows]
        block_start = document_start.unsqueeze(-1) + blocks * layout.compress_ratio
        tail_start = torch.maximum(document_start, positions - layout.compress_ratio + 1)
        starts = torch.cat([torch.where(kept, block_start, positions.unsqueeze(-1)), tail_start.unsqueeze(-1)], dim=1)
        touched = _touched_kv_blocks(starts, layout.compress_ratio, layout.kv_blocks)
        tiles = (positions - layout.query_offset) // FLEX_BLOCK_SIZE
        present[tiles[:, None, None].expand_as(touched), touched] = True

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        layout: QsaLayout,
        cp_group: dist.ProcessGroup | None,
        cp_world_size: int,
    ) -> QsaSelection:
        total_q = hidden_states.shape[1]
        query, keys = self.index_qk_proj(hidden_states).split(
            [self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim], dim=-1
        )
        cos, sin = position_embeddings
        query_slice = slice(layout.query_offset, layout.query_offset + total_q)
        query = self.q_layernorm(query.unflatten(-1, (self.num_heads, self.head_dim)))
        query = _apply_partial_rope(query, cos[:, query_slice].unsqueeze(2), sin[:, query_slice].unsqueeze(2))[0]

        if cp_world_size > 1:
            keys = gather_for_cp_wo_grad(keys.contiguous(), cp_world_size, cp_group)
        block_keys = self._pooled_block_keys(keys[0], layout, cos[0], sin[0])

        document_q = layout.document_q[:total_q]
        scored_blocks = layout.scored_blocks[:total_q]
        present = query.new_zeros(layout.q_blocks, layout.kv_blocks, dtype=torch.bool)
        topk = min(self.block_topk, block_keys.shape[0])
        selected = _select_blocks(
            query,
            block_keys,
            document_q,
            scored_blocks,
            layout.block_document,
            layout.block_index,
            topk,
            self.scaling,
        )
        for start in range(0, total_q, INDEXER_QUERY_CHUNK):
            rows = slice(start, min(start + INDEXER_QUERY_CHUNK, total_q))
            blocks = selected[rows]
            kept = blocks != _NO_BLOCK
            self._mark_reachable(present, layout, rows, blocks, kept)
        padded_rows = layout.q_blocks * FLEX_BLOCK_SIZE - total_q
        if padded_rows:
            selected = F.pad(selected, (0, 0, 0, padded_rows), value=_NO_BLOCK)
        return QsaSelection(selected, *_ordered_kv_blocks(present))


class Qwen4ExpSparseAttention(Qwen3_5MoeGatedAttentionBase):
    _varlen_funcs = {2: flash_attn_varlen_func, 3: flash_attn_3_varlen_func, 4: flash_attn_4_varlen_func}

    def __init__(self, config: Qwen4ExpConfig, flash_attn_version: int = 4):
        super().__init__(config)
        self.indexer = Qwen4ExpQSAIndexer(config)
        self.mask_mod = QsaMaskMod()
        self._flash_attn_version = flash_attn_version
        varlen_func = self._varlen_funcs[flash_attn_version]
        self._flash_attn_call = torch._dynamo.disable(varlen_func) if flash_attn_version == 4 else varlen_func
        self._cp_group: dist.ProcessGroup | None = None
        self._cp_rank = 0
        self._cp_world_size = 1

    def _attn_projections(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape))
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value = self.v_proj(hidden_states).view(hidden_shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query.transpose(1, 2), key.transpose(1, 2), value, gate

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def _dense_attention(self, query: Tensor, key: Tensor, value: Tensor, cu_seqlens: Tensor, max_seqlen: int):
        kwargs: dict = {"causal": True}
        if self._flash_attn_version == 4:
            kwargs["cu_seqlens_q"] = cu_seqlens
            kwargs["cu_seqlens_k"] = cu_seqlens
            out = self._flash_attn_call(query, key, value, **kwargs)
        else:
            out = self._flash_attn_call(query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def _sparse_attention(self, query: Tensor, key: Tensor, value: Tensor, layout: QsaLayout, selection: QsaSelection):
        self.mask_mod.update(layout, selection.selected_blocks)
        block_mask = BlockMask.from_kv_blocks(
            selection.kv_num_blocks,
            selection.kv_indices,
            BLOCK_SIZE=FLEX_BLOCK_SIZE,
            mask_mod=self.mask_mod,
            seq_lengths=layout.seq_lengths,
        )
        return _flex_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            block_mask=block_mask,
            scale=self.scaling,
            enable_gqa=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        qsa_layout: QsaLayout | None = None,
    ) -> tuple[Tensor, None]:
        cos, sin = position_embeddings
        local = slice(self._cp_rank * hidden_states.shape[1], (self._cp_rank + 1) * hidden_states.shape[1])
        query, key, value, gate = self._attn_projections(hidden_states, (cos[:, local], sin[:, local]))

        if qsa_layout is None:
            return (
                self.output_proj(self._dense_attention(query[0], key[0], value[0], cu_seqlens, max_seqlen), gate),
                None,
            )

        selection = self.indexer(hidden_states, position_embeddings, qsa_layout, self._cp_group, self._cp_world_size)
        if self.cp_enabled:
            key = gather_for_cp(key, self._cp_group)
            value = gather_for_cp(value, self._cp_group)
        return self.output_proj(self._sparse_attention(query, key, value, qsa_layout, selection), gate), None

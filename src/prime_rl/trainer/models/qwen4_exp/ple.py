import math

import torch
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
from torch import nn

from .configuration_qwen4_exp import Qwen4ExpConfig
from .ngram_table import PendingNGramLookup, ShardedNGramTable
from .norms import Qwen4ExpRMSNorm

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _first_eos(config: Qwen4ExpConfig) -> int:
    eos_token_id = config.eos_token_id
    if eos_token_id is None:
        raise ValueError("Qwen4-Exp per-layer embeddings need an `eos_token_id` to reset the n-gram context")
    return eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> torch.Tensor:
    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64) % half_bound) + 1
        for index in range(ngram_size)
    ]
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def first_primes_after(start: int, count: int) -> list[int]:
    primes: list[int] = []
    candidate = start
    while len(primes) < count:
        candidate += 1
        if _is_prime(candidate):
            primes.append(candidate)
    return primes


def _left_halo(x: torch.Tensor, width: int, fill: float, cp_group, cp_rank: int) -> torch.Tensor:
    padding = x.new_full((x.shape[0], width, *x.shape[2:]), fill)
    if cp_group is None:
        return padding
    if x.shape[1] < width:
        raise ValueError(f"context-parallel shard of {x.shape[1]} tokens is shorter than the PLE halo of {width}")
    tails = dist_nn.all_gather(x[:, -width:].contiguous(), group=cp_group)
    return torch.stack([padding, *tails[:-1]])[cp_rank]


def _document_starts(positions: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    bounds = cu_seqlens.to(positions.dtype)
    return bounds[torch.searchsorted(bounds, positions.clamp_min(0), right=True) - 1]


def _previous_eos(token_ids: torch.Tensor, positions: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    eos_at = torch.where(token_ids == eos_token_id, positions, positions.new_full((), -1))
    return F.pad(torch.cummax(eos_at, dim=-1).values[:, :-1], (1, 0), value=-1)


def _dilated_causal_conv(
    hidden_states: torch.Tensor, halo: torch.Tensor, weight: torch.Tensor, dilation: int, reach: torch.Tensor
) -> torch.Tensor:
    kernel_size = weight.shape[-1]
    span = (kernel_size - 1) * dilation
    history = torch.cat([halo, hidden_states], dim=1)
    output = torch.zeros_like(hidden_states)
    for tap in range(kernel_size):
        shift = (kernel_size - 1 - tap) * dilation
        window = history[:, span - shift : history.shape[1] - shift]
        output = output + window * weight[:, 0, tap] * (reach >= shift)
    return output


class Qwen4ExpNGramEmbedding(nn.Module):
    layer_multipliers: torch.Tensor
    ngram_heads_vocab_sizes: torch.Tensor
    ngram_heads_offsets: torch.Tensor

    def __init__(self, config: Qwen4ExpConfig, ple_layer_index: int):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.seed = config.seed
        self.initializer_range = config.initializer_range
        self.eos_token_id = _first_eos(config)
        self.head_dim = config.ple_embed_dim // self.ngram_heads

        head_count = (ple_layer_index + 1) * self.ngram_heads
        self.head_vocab_sizes = first_primes_after(config.ngram_vocab_size_base - 1, head_count)[
            ple_layer_index * self.ngram_heads :
        ]
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(sum(self.head_vocab_sizes) / divisor) * divisor
        if padded_vocab_size % config.split_ngram_parts:
            raise ValueError(
                f"n-gram vocabulary of {padded_vocab_size} does not split evenly into "
                f"{config.split_ngram_parts} shards; make_ngram_vocab_size_divisible_by must be a multiple of it"
            )

        self.register_buffer("layer_multipliers", torch.empty(self.ngram_size, dtype=torch.long))
        self.register_buffer("ngram_heads_vocab_sizes", torch.empty(self.ngram_heads, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.empty(self.ngram_heads, dtype=torch.long))
        self.ngram_embedding = nn.Parameter(
            torch.empty(config.split_ngram_parts, padded_vocab_size // config.split_ngram_parts, self.head_dim),
            requires_grad=False,
        )
        self.external_table: ShardedNGramTable | None = None
        self.reset_buffers()

    def externalize(self) -> None:
        table = self.ngram_embedding
        self.external_table = ShardedNGramTable(
            table.shape[0], table.shape[1], table.shape[2], table.dtype, self.initializer_range, self.seed
        )
        del self.ngram_embedding

    def bind_external_table(self, snapshot_path, prefix: str, group=None) -> None:
        if self.external_table is None:
            raise RuntimeError("Qwen4-Exp PLE table was not externalized")
        self.external_table.bind(snapshot_path, prefix, group)

    def reset_buffers(self) -> None:
        sizes = torch.tensor(self.head_vocab_sizes, dtype=torch.long)
        self.layer_multipliers.copy_(
            build_layer_multipliers(self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed)
        )
        self.ngram_heads_vocab_sizes.copy_(sizes)
        self.ngram_heads_offsets.copy_(F.pad(sizes.cumsum(0)[:-1], (1, 0)))

    def forward(self, history: torch.Tensor, reach: torch.Tensor) -> torch.Tensor | PendingNGramLookup:
        context_len = self.ngram_size - 1
        seq_len = history.shape[1] - context_len
        shifted = [
            torch.where(
                reach >= shift,
                history[:, context_len - shift : context_len - shift + seq_len],
                history.new_full((), self.eos_token_id),
            )
            for shift in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            heads = slice((ngram - 2) * self.heads_per_ngram, (ngram - 1) * self.heads_per_ngram)
            mixed_ids = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted[position] * self.layer_multipliers[position])
            bucket = torch.remainder(mixed_ids.unsqueeze(-1), self.ngram_heads_vocab_sizes[heads])
            blocks.append(bucket + self.ngram_heads_offsets[heads])

        ngram_ids = torch.cat(blocks, dim=-1)
        if self.external_table is not None:
            return self.external_table.start(ngram_ids)
        embedding_device = self.ngram_embedding.device
        embeddings = F.embedding(ngram_ids.to(embedding_device), self.ngram_embedding.view(-1, self.head_dim)).flatten(
            -2
        )
        return embeddings.to(ngram_ids.device)


class Qwen4ExpPLELayer(nn.Module):
    def __init__(self, config: Qwen4ExpConfig, ple_layer_index: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.ngram_size = config.ngram_size
        self.eos_token_id = _first_eos(config)
        hc_hidden_size = config.hidden_size * config.hc_count

        self.ple_embedding = Qwen4ExpNGramEmbedding(config, ple_layer_index)
        self.key_proj = nn.Linear(config.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(config.ple_embed_dim, config.hidden_size, bias=False)
        self.norm_key = Qwen4ExpRMSNorm(hc_hidden_size, group_size=config.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpRMSNorm(hc_hidden_size, group_size=config.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpRMSNorm(hc_hidden_size, group_size=config.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=config.ple_conv_kernel_size,
            groups=hc_hidden_size,
            dilation=config.ngram_size,
            bias=False,
        )
        self.cp_group = None
        self.cp_rank = 0

    def set_context_parallel_attributes(self, cp_group, cp_rank: int) -> None:
        self.cp_group = cp_group
        self.cp_rank = cp_rank

    def _gate(self, embeddings: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        key = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        query = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        alignment = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        signed_root = alignment.abs().clamp_min(1e-6).sqrt() * alignment.sign()
        value = self.value_proj(embeddings).unsqueeze(-2)
        return (torch.sigmoid(signed_root) * value).flatten(-2)

    def prepare(
        self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> tuple[torch.Tensor | PendingNGramLookup, torch.Tensor]:
        input_ids = input_ids.long()
        seq_len = input_ids.shape[1]
        context_len = self.ngram_size - 1
        offset = self.cp_rank * seq_len

        id_history = torch.cat(
            [_left_halo(input_ids, context_len, self.eos_token_id, self.cp_group, self.cp_rank), input_ids], dim=1
        )
        positions = torch.arange(offset - context_len, offset + seq_len, device=input_ids.device)
        document_start = _document_starts(positions, cu_seqlens)
        segment_start = torch.maximum(document_start, _previous_eos(id_history, positions, self.eos_token_id) + 1)
        ngram_reach = (positions - segment_start)[:, context_len:]
        conv_reach = (positions - document_start)[context_len:].view(1, -1, 1)
        return self.ple_embedding(id_history, ngram_reach), conv_reach

    def resolve(
        self, prepared: tuple[torch.Tensor | PendingNGramLookup, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings, conv_reach = prepared
        if isinstance(embeddings, PendingNGramLookup):
            embeddings = embeddings.result().flatten(-2)
        return embeddings, conv_reach

    def forward(self, hidden_states: torch.Tensor, prepared: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        embeddings, conv_reach = prepared
        embeddings = embeddings.to(hidden_states.dtype)
        span = (self.conv1d.kernel_size[0] - 1) * self.conv1d.dilation[0]
        gated_value = self._gate(embeddings, hidden_states)
        conv_input = self.norm_conv(gated_value)
        conv_halo = _left_halo(conv_input, span, 0.0, self.cp_group, self.cp_rank)
        return gated_value + F.silu(
            _dilated_causal_conv(conv_input, conv_halo, self.conv1d.weight, self.conv1d.dilation[0], conv_reach)
        )

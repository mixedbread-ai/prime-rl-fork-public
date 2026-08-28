from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import Tensor

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


class PendingNGramLookup:
    def __init__(
        self,
        future: Future[Tensor],
        shape: tuple[int, ...],
        table: ShardedNGramTable,
        send_counts: list[int] | None = None,
        recv_counts: list[int] | None = None,
        inverse: Tensor | None = None,
    ):
        self.future = future
        self.shape = shape
        self.table = table
        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.inverse = inverse

    def done(self) -> bool:
        return self.future.done()

    def result(self) -> Tensor:
        embeddings = self.future.result()
        if self.inverse is None:
            return embeddings.view(self.shape)
        sorted_embeddings = embeddings.new_empty(self.inverse.numel(), self.table.head_dim)
        dist.all_to_all_single(
            sorted_embeddings,
            embeddings,
            output_split_sizes=self.send_counts,
            input_split_sizes=self.recv_counts,
            group=self.table.group,
        )
        return sorted_embeddings[self.inverse].view(self.shape)


class ShardedNGramTable:
    def __init__(
        self, shard_count: int, rows_per_shard: int, head_dim: int, dtype: torch.dtype, init_std: float, seed: int
    ):
        self.shard_count = shard_count
        self.rows_per_shard = rows_per_shard
        self.head_dim = head_dim
        self.init_std = init_std
        self.seed = seed
        self.group = None
        self.rank = 0
        self.world_size = 1
        self.dtype = dtype
        self.shards: dict[int, Tensor] = {}
        self.executor: ThreadPoolExecutor | None = None
        self.copy_stream: torch.cuda.Stream | None = None

    def _owner(self, shard: Tensor) -> Tensor:
        return shard * self.world_size // self.shard_count

    def _owned_shards(self) -> list[int]:
        return [i for i in range(self.shard_count) if i * self.world_size // self.shard_count == self.rank]

    @staticmethod
    def _weight_map(snapshot_path: Path) -> tuple[Path, dict[str, str]]:
        manifest_path = snapshot_path / "external_weights.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            source = (snapshot_path / manifest["source"]).resolve()
            return source, manifest["weight_map"]
        index_path = snapshot_path / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            if weight_map:
                return snapshot_path, weight_map
        weight_map = {}
        for path in snapshot_path.glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as reader:
                weight_map.update(dict.fromkeys(reader.keys(), path.name))
        return snapshot_path, weight_map

    def _validate_manifest(self, snapshot_path: Path, prefix: str, weight_map: dict[str, str]) -> torch.dtype:
        files: dict[str, list[str]] = {}
        for shard in range(self.shard_count):
            key = f"{prefix}.shard_{shard}.weight"
            filename = weight_map.get(key)
            if filename is None:
                raise KeyError(f"Qwen4-Exp checkpoint is missing {key}")
            files.setdefault(filename, []).append(key)

        dtype = None
        expected_shape = [self.rows_per_shard, self.head_dim]
        for filename, keys in files.items():
            with safe_open(snapshot_path / filename, framework="pt", device="cpu") as reader:
                for key in keys:
                    tensor = reader.get_slice(key)
                    if tensor.get_shape() != expected_shape:
                        raise ValueError(
                            f"Qwen4-Exp checkpoint has shape {tensor.get_shape()} for {key}; expected {expected_shape}"
                        )
                    tensor_dtype = tensor.get_dtype()
                    if dtype is not None and tensor_dtype != dtype:
                        raise ValueError(f"Qwen4-Exp PLE shards have mixed dtypes: {dtype} and {tensor_dtype}")
                    dtype = tensor_dtype
        if dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"unsupported Qwen4-Exp PLE dtype: {dtype}")
        return _SAFETENSORS_DTYPES[dtype]

    def bind(self, snapshot_path: Path | None, prefix: str, group=None) -> None:
        self.group = group
        if dist.is_initialized() and group is not None:
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        owned = self._owned_shards()
        if snapshot_path is None:
            self.shards = {}
            for shard in owned:
                generator = torch.Generator().manual_seed(self.seed + shard)
                self.shards[shard] = torch.empty(self.rows_per_shard, self.head_dim, dtype=self.dtype).normal_(
                    std=self.init_std, generator=generator
                )
            return

        snapshot_path, weight_map = self._weight_map(snapshot_path)
        snapshot_path = snapshot_path.resolve()
        self.dtype = self._validate_manifest(snapshot_path, prefix, weight_map)
        files: dict[str, list[tuple[int, str]]] = {}
        for shard in owned:
            key = f"{prefix}.shard_{shard}.weight"
            filename = weight_map.get(key)
            files.setdefault(filename, []).append((shard, key))
        self.shards = {}
        for filename, entries in files.items():
            with safe_open(snapshot_path / filename, framework="pt", device="cpu") as reader:
                for shard, key in entries:
                    self.shards[shard] = reader.get_tensor(key)

    def _lookup_cpu(self, ids: Tensor, pin_memory: bool = False) -> Tensor:
        shard_ids = torch.div(ids, self.rows_per_shard, rounding_mode="floor")
        rows = torch.remainder(ids, self.rows_per_shard)
        output = torch.empty(ids.numel(), self.head_dim, dtype=self.dtype, pin_memory=pin_memory)
        for shard in shard_ids.unique().tolist():
            mask = shard_ids == shard
            output[mask] = self.shards[shard].index_select(0, rows[mask])
        return output

    def _lookup_staged(self, ids: Tensor, output: Tensor, stream: torch.cuda.Stream, ready: torch.cuda.Event) -> Tensor:
        ready.synchronize()
        host_output = self._lookup_cpu(ids, pin_memory=True)
        with torch.cuda.device(output.device), torch.cuda.stream(stream):
            output.copy_(host_output, non_blocking=True)
            copied = stream.record_event()
        copied.synchronize()
        return output

    def _start_local(self, ids: Tensor) -> Future[Tensor]:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1)
        if not ids.is_cuda:
            return self.executor.submit(self._lookup_cpu, ids)
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device=ids.device)
        cpu_ids = torch.empty_like(ids, device="cpu", pin_memory=True)
        output = torch.empty(ids.numel(), self.head_dim, dtype=self.dtype, device=ids.device)
        self.copy_stream.wait_stream(torch.cuda.current_stream(ids.device))
        with torch.cuda.stream(self.copy_stream):
            cpu_ids.copy_(ids, non_blocking=True)
            ready = self.copy_stream.record_event()
        ids.record_stream(self.copy_stream)
        return self.executor.submit(self._lookup_staged, cpu_ids, output, self.copy_stream, ready)

    def start(self, ids: Tensor) -> PendingNGramLookup:
        shape = (*ids.shape, self.head_dim)
        flat_ids = ids.flatten()
        if self.world_size == 1:
            return PendingNGramLookup(self._start_local(flat_ids), shape, self)

        shards = torch.div(flat_ids, self.rows_per_shard, rounding_mode="floor")
        owners = self._owner(shards)
        order = owners.argsort(stable=True)
        send_ids = flat_ids[order].contiguous()
        send_counts = torch.bincount(owners, minlength=self.world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)

        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        recv_ids = flat_ids.new_empty(sum(recv_counts_list))
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list,
            group=self.group,
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return PendingNGramLookup(self._start_local(recv_ids), shape, self, send_counts_list, recv_counts_list, inverse)

    def __call__(self, ids: Tensor) -> Tensor:
        return self.start(ids).result()

import torch

from compact_asm_cache import compact_cache_file
from datasets.asm_encoding import _save_tensor_atomic


def test_save_tensor_atomic_does_not_serialize_unused_view_storage(tmp_path):
    chunk = torch.arange(4096 * 768, dtype=torch.float32).reshape(4096, 768)
    pooled_view = chunk[123:124]
    cache_path = tmp_path / "pooled.pt"

    _save_tensor_atomic(pooled_view, cache_path)

    loaded = torch.load(cache_path, weights_only=True)
    assert torch.equal(loaded, pooled_view)
    assert cache_path.stat().st_size < 16 * 1024


def test_compact_cache_file_rewrites_oversized_view_storage(tmp_path):
    chunk = torch.arange(4096 * 768, dtype=torch.float32).reshape(4096, 768)
    pooled_view = chunk[123:124]
    cache_path = tmp_path / "pooled.pt"
    torch.save(pooled_view, cache_path)
    oversized_size = cache_path.stat().st_size

    result = compact_cache_file(cache_path)

    loaded = torch.load(cache_path, weights_only=True)
    assert result.compacted
    assert result.size_before == oversized_size
    assert result.size_after < 16 * 1024
    assert torch.equal(loaded, pooled_view)

    second_result = compact_cache_file(cache_path)
    assert not second_result.compacted
    assert second_result.size_before == second_result.size_after

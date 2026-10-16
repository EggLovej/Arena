"""Which device learns: an NVIDIA GPU (e.g. on Colab), the Apple GPU (this Mac), or the CPU."""

import torch


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# How much the GPU may hold, cache included, before release_cached_memory gives the cache back.
# One round of self-play learning uses about 3.3 GB.
CACHE_LIMIT = 4 * 2**30


def release_cached_memory(device: torch.device | str, limit: int = CACHE_LIMIT) -> None:
    """Give the GPU's cache of freed buffers back to the system once the GPU holds more than ``limit`` bytes.

    PyTorch keeps freed GPU buffers to use again. Batches from maps of different sizes need
    buffers of different sizes, and the cache keeps them all, so a long run grew by gigabytes
    (history.md, "What the memory that builds up actually is"). Releasing has a price: the next
    round allocates its buffers afresh, which made learning about 30% slower when it was done
    every round. So the cache is only released once it is large. ``limit=0`` always releases.
    On the CPU there is nothing to do.
    """

    kind = torch.device(device).type
    if kind == "mps" and torch.mps.driver_allocated_memory() > limit:
        torch.mps.empty_cache()
    elif kind == "cuda" and torch.cuda.memory_reserved() > limit:
        torch.cuda.empty_cache()

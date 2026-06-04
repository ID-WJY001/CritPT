#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2
    if rank == 0:
        print(f"all_reduce={value.item()} expected={expected}")
    assert value.item() == expected
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()


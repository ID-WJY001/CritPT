#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("RL_DATA_ROOT", "/data/sdb/rl-posttrain"))
SERVER_TARGET = ROOT / "repos/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
UTILS_TARGET = ROOT / "repos/verl/verl/workers/rollout/vllm_rollout/utils.py"
RAY_TRAINER_TARGET = ROOT / "repos/verl/verl/trainer/ppo/ray_trainer.py"
ATTENTION_TARGET = ROOT / "repos/verl/verl/utils/attention_utils.py"
TORCH_FUNCTIONAL_TARGET = ROOT / "repos/verl/verl/utils/torch_functional.py"

OLD = "from vllm.entrypoints.cli.serve import run_headless\n"
NEW = """try:
    from vllm.entrypoints.cli.serve import run_headless
except ImportError:
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server

    def run_headless(args):
        return uvloop.run(run_server(args))
"""

OLD_LOGPROBS = '            "logprobs_mode": self.config.logprobs_mode,\n'
NEW_LOGPROBS = (
    '            **({"logprobs_mode": self.config.logprobs_mode} '
    'if _VLLM_VERSION >= version.parse("0.10.0") else {}),\n'
)

OLD_RESET_MM_CACHE = """        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )
"""
NEW_RESET_MM_CACHE = """        # Don't keep the dummy data in memory. vLLM 0.8.x V1 AsyncLLM
        # does not expose reset_mm_cache(), while newer versions do.
        if hasattr(engine_client, "reset_mm_cache"):
            await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )
"""

OLD_WAIT_FOR_DRAIN = """    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()
"""
NEW_WAIT_FOR_DRAIN = """    async def wait_for_requests_to_drain(self):
        if hasattr(self.engine, "wait_for_requests_to_drain"):
            await self.engine.wait_for_requests_to_drain()
"""

OLD_EMPTY_MM_PROMPT = """        prompt_kwargs = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}
        if mm_processor_kwargs:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
"""
NEW_EMPTY_MM_PROMPT = """        prompt_kwargs = {"prompt_token_ids": prompt_ids}
        if multi_modal_data:
            prompt_kwargs["multi_modal_data"] = multi_modal_data
        if mm_processor_kwargs:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
"""

OLD_PROCESS_WEIGHTS = """        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            for model, model_config in self._iter_all_models_with_config():
                process_weights_after_loading(model, model_config, self.device)
"""
NEW_PROCESS_WEIGHTS = """        elif use_standard_weight_load:
            # vLLM 0.8.x does not export process_weights_after_loading.
            # Plain bf16 Qwen3 weights do not need the post-load quantization hook.
            try:
                from vllm.model_executor.model_loader.utils import process_weights_after_loading
            except ImportError:
                process_weights_after_loading = None

            if process_weights_after_loading is not None:
                for model, model_config in self._iter_all_models_with_config():
                    process_weights_after_loading(model, model_config, self.device)
"""

OLD_DROP_REWARD_KEYS = """                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
"""
NEW_DROP_REWARD_KEYS = """                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # Async agent rollout may echo input-side reward metadata back in
                    # gen_batch_output, sometimes in completion order. Keep the original
                    # repeated batch metadata as the source of truth before union.
                    input_reward_keys = {"data_source", "reward_model", "extra_info", "uid"}
                    conflict_non_tensor_keys = list(input_reward_keys & gen_batch_output.non_tensor_batch.keys())
                    if conflict_non_tensor_keys:
                        gen_batch_output.pop(non_tensor_batch_keys=conflict_non_tensor_keys)
                    batch = batch.union(gen_batch_output)
"""

OLD_DROP_VALIDATE_KEYS = """            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
"""
NEW_DROP_VALIDATE_KEYS = """            # Async validation rollout can echo input-side reward metadata back
            # in completion order. Keep test_batch metadata as the source of truth.
            input_reward_keys = {"data_source", "reward_model", "extra_info", "uid"}
            conflict_non_tensor_keys = list(input_reward_keys & test_output_gen_batch.non_tensor_batch.keys())
            if conflict_non_tensor_keys:
                test_output_gen_batch.pop(non_tensor_batch_keys=conflict_non_tensor_keys)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
"""

OLD_ATTENTION_FALLBACK = """    else:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
"""
NEW_ATTENTION_FALLBACK = """    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            import torch
            import torch.nn.functional as F
            from einops import rearrange

            def index_first_axis(input, indices):
                return input[indices]

            def pad_input(hidden_states, indices, batch, seqlen):
                output = torch.zeros(
                    (batch * seqlen, *hidden_states.shape[1:]),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                output[indices] = hidden_states
                return rearrange(output, "(b s) ... -> b s ...", b=batch)

            def unpad_input(hidden_states, attention_mask, unused_mask=None):
                seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
                indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                max_seqlen_in_batch = seqlens_in_batch.max().item()
                cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
                return (
                    index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
                    indices,
                    cu_seqlens,
                    max_seqlen_in_batch,
                )
"""

OLD_TORCH_FUNCTIONAL_UNPAD_IMPORT = "    from flash_attn.bert_padding import pad_input, unpad_input\n"
NEW_TORCH_FUNCTIONAL_UNPAD_IMPORT = "    from verl.utils.attention_utils import pad_input, unpad_input\n"

OLD_TORCH_FUNCTIONAL_PAD_IMPORT = """    if get_device_name() == "cuda":
        from flash_attn.bert_padding import pad_input
    elif get_device_name() == "npu":
        from verl.utils.attention_utils import pad_input
"""
NEW_TORCH_FUNCTIONAL_PAD_IMPORT = """    from verl.utils.attention_utils import pad_input
"""


def patch_server() -> bool:
    if not SERVER_TARGET.exists():
        raise SystemExit(f"verl vLLM server file not found: {SERVER_TARGET}")

    text = SERVER_TARGET.read_text(encoding="utf-8")
    changed = False

    if "def run_headless(args):\n        return uvloop.run(run_server(args))" not in text:
        if OLD not in text:
            raise SystemExit(f"expected import not found in {SERVER_TARGET}")
        text = text.replace(OLD, NEW)
        changed = True

    if OLD_LOGPROBS in text:
        text = text.replace(OLD_LOGPROBS, NEW_LOGPROBS)
        changed = True

    if OLD_RESET_MM_CACHE in text:
        text = text.replace(OLD_RESET_MM_CACHE, NEW_RESET_MM_CACHE)
        changed = True

    if OLD_WAIT_FOR_DRAIN in text:
        text = text.replace(OLD_WAIT_FOR_DRAIN, NEW_WAIT_FOR_DRAIN)
        changed = True

    if OLD_EMPTY_MM_PROMPT in text:
        text = text.replace(OLD_EMPTY_MM_PROMPT, NEW_EMPTY_MM_PROMPT)
        changed = True

    if changed:
        SERVER_TARGET.write_text(text, encoding="utf-8")
        print(f"patched: {SERVER_TARGET}")
    else:
        print(f"already patched: {SERVER_TARGET}")
    return changed


def patch_utils() -> bool:
    if not UTILS_TARGET.exists():
        raise SystemExit(f"verl vLLM utils file not found: {UTILS_TARGET}")

    text = UTILS_TARGET.read_text(encoding="utf-8")
    if OLD_PROCESS_WEIGHTS in text:
        text = text.replace(OLD_PROCESS_WEIGHTS, NEW_PROCESS_WEIGHTS)
        UTILS_TARGET.write_text(text, encoding="utf-8")
        print(f"patched: {UTILS_TARGET}")
        return True

    if "process_weights_after_loading = None" in text:
        print(f"already patched: {UTILS_TARGET}")
        return False

    raise SystemExit(f"expected process_weights_after_loading block not found in {UTILS_TARGET}")


def patch_ray_trainer() -> bool:
    if not RAY_TRAINER_TARGET.exists():
        raise SystemExit(f"verl Ray trainer file not found: {RAY_TRAINER_TARGET}")

    text = RAY_TRAINER_TARGET.read_text(encoding="utf-8")
    changed = False

    if OLD_DROP_REWARD_KEYS in text:
        text = text.replace(OLD_DROP_REWARD_KEYS, NEW_DROP_REWARD_KEYS)
        changed = True

    if OLD_DROP_VALIDATE_KEYS in text:
        text = text.replace(OLD_DROP_VALIDATE_KEYS, NEW_DROP_VALIDATE_KEYS)
        changed = True

    if changed:
        RAY_TRAINER_TARGET.write_text(text, encoding="utf-8")
        print(f"patched: {RAY_TRAINER_TARGET}")
        return True

    if "gen_batch_output.pop(non_tensor_batch_keys=conflict_non_tensor_keys)" in text and (
        "test_output_gen_batch.pop(non_tensor_batch_keys=conflict_non_tensor_keys)" in text
    ):
        print(f"already patched: {RAY_TRAINER_TARGET}")
        return False

    raise SystemExit(f"expected batch union blocks not found in {RAY_TRAINER_TARGET}")


def patch_attention_utils() -> bool:
    if not ATTENTION_TARGET.exists():
        raise SystemExit(f"verl attention utils file not found: {ATTENTION_TARGET}")

    text = ATTENTION_TARGET.read_text(encoding="utf-8")
    if OLD_ATTENTION_FALLBACK in text:
        text = text.replace(OLD_ATTENTION_FALLBACK, NEW_ATTENTION_FALLBACK)
        ATTENTION_TARGET.write_text(text, encoding="utf-8")
        print(f"patched: {ATTENTION_TARGET}")
        return True

    if "except ImportError:" in text and "def index_first_axis(input, indices):" in text:
        print(f"already patched: {ATTENTION_TARGET}")
        return False

    raise SystemExit(f"expected flash-attn import block not found in {ATTENTION_TARGET}")


def patch_torch_functional() -> bool:
    if not TORCH_FUNCTIONAL_TARGET.exists():
        raise SystemExit(f"verl torch_functional file not found: {TORCH_FUNCTIONAL_TARGET}")

    text = TORCH_FUNCTIONAL_TARGET.read_text(encoding="utf-8")
    changed = False

    if OLD_TORCH_FUNCTIONAL_UNPAD_IMPORT in text:
        text = text.replace(OLD_TORCH_FUNCTIONAL_UNPAD_IMPORT, NEW_TORCH_FUNCTIONAL_UNPAD_IMPORT, 1)
        changed = True

    if OLD_TORCH_FUNCTIONAL_PAD_IMPORT in text:
        text = text.replace(OLD_TORCH_FUNCTIONAL_PAD_IMPORT, NEW_TORCH_FUNCTIONAL_PAD_IMPORT)
        changed = True

    if changed:
        TORCH_FUNCTIONAL_TARGET.write_text(text, encoding="utf-8")
        print(f"patched: {TORCH_FUNCTIONAL_TARGET}")
        return True

    if "from verl.utils.attention_utils import pad_input, unpad_input" in text:
        print(f"already patched: {TORCH_FUNCTIONAL_TARGET}")
        return False

    raise SystemExit(f"expected flash-attn import blocks not found in {TORCH_FUNCTIONAL_TARGET}")


def main() -> None:
    patch_server()
    patch_utils()
    patch_ray_trainer()
    patch_attention_utils()
    patch_torch_functional()


if __name__ == "__main__":
    main()

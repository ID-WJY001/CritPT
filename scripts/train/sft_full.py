#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass
class SFTItem:
    prompt: str
    response: str


class ChatSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_length: int):
        self.items = _read_items(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.items[index]
        user_messages = [{"role": "user", "content": item.prompt}]
        full_messages = [
            {"role": "user", "content": item.prompt},
            {"role": "assistant", "content": item.response},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


@dataclass
class CausalCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            labels.append(feature["labels"] + [-100] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _read_items(path: Path) -> list[SFTItem]:
    items: list[SFTItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages: list[dict[str, Any]] = row["messages"]
            prompt = next(msg["content"] for msg in messages if msg["role"] == "user")
            response = next(msg["content"] for msg in messages if msg["role"] == "assistant")
            items.append(SFTItem(prompt=prompt, response=response))
    return items


def _print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    pct = 100 * trainable / total
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fsdp-transformer-layer", default="Qwen3DecoderLayer")
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="no")
    parser.add_argument("--skip-final-save", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    _print_trainable_parameters(model)

    train_dataset = ChatSFTDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ChatSFTDataset(args.val_data, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=args.save_strategy,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": args.fsdp_transformer_layer,
            "activation_checkpointing": True,
            "use_orig_params": True,
            "state_dict_type": "FULL_STATE_DICT",
        },
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    if not args.skip_final_save:
        trainer.save_model(str(args.output_dir / "final_model"))
        tokenizer.save_pretrained(str(args.output_dir / "final_model"))


if __name__ == "__main__":
    main()

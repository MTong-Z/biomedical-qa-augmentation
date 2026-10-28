import json
from pathlib import Path

from .io import file_hash, read_jsonl, write_json
from .model import render_prompt


def encode_example(row, tokenizer, max_length):
    prompt = render_prompt(tokenizer, row["prompt"])
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(json.dumps(row["response"], ensure_ascii=False), add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(suffix) > max_length:
        raise ValueError(f"Training record {row['id']} exceeds max_length; no silent truncation")
    return {"input_ids": prefix + suffix, "attention_mask": [1] * (len(prefix) + len(suffix)), "labels": [-100] * len(prefix) + suffix}


def train(train_path, dev_path, model_name, revision, output, steps=100, seed=42, max_length=4096, four_bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError("Run training inside a GPU allocation; CUDA is not available")
    if steps <= 0:
        raise ValueError("Training steps must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty training output directory")
    set_seed(seed)
    rows, dev = read_jsonl(train_path), read_jsonl(dev_path)
    if not rows or not dev or {r['source_id'] for r in rows} & {r['source_id'] for r in dev}:
        raise ValueError("Empty or overlapping training and development data")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [encode_example(row, tokenizer, max_length) for row in rows]
    encoded_dev = [encode_example(row, tokenizer, max_length) for row in dev]
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    options = {"revision": revision, "trust_remote_code": False, "torch_dtype": dtype, "device_map": {"": 0}}
    if four_bit:
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, **options)
    if four_bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        result = {}
        for key, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in batch], dtype=torch.long)
        return result

    interval = max(1, steps // 2)
    arguments = TrainingArguments(output_dir=str(output), max_steps=steps, learning_rate=2e-4,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                                  warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type="cosine", gradient_checkpointing=True,
                                  bf16=bf16, fp16=not bf16, logging_steps=1, eval_strategy="steps", eval_steps=interval,
                                  save_strategy="steps", save_steps=interval, save_total_limit=2, load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss", greater_is_better=False, report_to=[], seed=seed, data_seed=seed)
    write_json(output / "run_manifest.json", {"base_model": model_name, "requested_revision": revision,
               "resolved_revision": getattr(model.config, "_commit_hash", None), "train_sha256": file_hash(train_path),
               "dev_sha256": file_hash(dev_path), "train_examples": len(rows), "dev_examples": len(dev),
               "seed": seed, "steps": steps, "four_bit": four_bit, "max_length": max_length,
               "training_arguments": arguments.to_dict(), "dataset_input_tokens": sum(len(x["input_ids"]) for x in encoded),
               "dataset_supervised_tokens": sum(sum(y != -100 for y in x["labels"]) for x in encoded),
               "gpu": torch.cuda.get_device_name(0), "package_versions": {"torch": torch.__version__},
               "comparison_protocol": "Same optimization steps and batch settings. Dataset lengths differ; this is not an exactly matched token budget."})
    trainer = Trainer(model=model, args=arguments, train_dataset=encoded, eval_dataset=encoded_dev, data_collator=collate, processing_class=tokenizer)
    outcome = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    write_json(output / "metrics.json", {"train": outcome.metrics, "validation": trainer.evaluate(), "best_checkpoint": trainer.state.best_model_checkpoint})
    return {"adapter": str(output / "adapter"), "global_step": trainer.state.global_step}

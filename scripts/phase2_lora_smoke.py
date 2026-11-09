"""Four-update LoRA smoke test: train, save, reload, then full-image val inference."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as metadata
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from phase2_smoke import DATA, MODEL_ID, OUT, infer_one, make_messages


SEED = 20260923
PROMPT = (
    "Based only on this thyroid ultrasound image, classify the nodule. "
    "Reply with exactly one word: benign or malignant."
)
STEPS = 4


def versions() -> dict[str, str]:
    names = ["torch", "transformers", "accelerate", "peft", "qwen-vl-utils", "Pillow"]
    return {name: metadata.version(name) for name in names}


def input_pair(processor, sample: dict):
    path = DATA / sample["image"]
    user_messages = make_messages(path)
    full_messages = user_messages + [{"role": "assistant", "content": sample["class"]}]
    user_text = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(user_messages)
    full_inputs = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    user_inputs = processor(
        text=[user_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    prefix_len = user_inputs["input_ids"].shape[-1]
    if not torch.equal(full_inputs["input_ids"][:, :prefix_len], user_inputs["input_ids"]):
        raise RuntimeError("Prompt tokens are not a prefix of the supervised conversation")
    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100
    if not (labels != -100).any():
        raise RuntimeError("No assistant target tokens remain after prompt masking")
    return full_inputs, labels


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "selection_seed20260923.json"
    selection = json.loads(selection_path.read_text())
    train_samples = selection["v1_train"]
    val_samples = [selection["v0_val"][0], selection["v0_val"][4]]
    if len(train_samples) != 4 or len(val_samples) != 2:
        raise ValueError("Expected four train and two validation smoke samples")

    adapter_dir = OUT / "v1_adapter_seed20260923"
    result_path = OUT / "v1_results.json"
    result = {
        "stage": "C / V1 LoRA smoke: four optimizer updates, save, reload, infer",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": SEED,
        "model_id": MODEL_ID,
        "packages": versions(),
        "gpu": torch.cuda.get_device_name(0),
        "train_samples": train_samples,
        "val_samples_after_reload": val_samples,
        "test_used": False,
        "exact_duplicate_groups_between_selected_train_and_val": selection[
            "exact_duplicate_groups_between_selected_train_and_val"
        ],
        "prompt_ground_truth_free": True,
        "prompt": PROMPT,
        "updates_requested": STEPS,
        "optimizer_updates": [],
        "adapter_path": str(adapter_dir),
        "validation_results": [],
    }
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        load_start = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": 0},
            local_files_only=True,
        )
        result["base_reload_seconds"] = round(time.perf_counter() - load_start, 3)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("PEFT attached no trainable parameters")
        result["trainable_parameters"] = sum(p.numel() for p in trainable)
        result["total_parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        for step in range(STEPS):
            sample = train_samples[step % len(train_samples)]
            started = time.perf_counter()
            record = {
                "step": step + 1,
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "training_target_for_supervision_only": sample["class"],
                "input": "original full image; assistant target only in labels",
            }
            try:
                inputs, labels = input_pair(processor, sample)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(**inputs, labels=labels)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {loss.detach().item()}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                record.update({
                    "status": "success",
                    "loss": round(float(loss.detach().cpu()), 6),
                    "grad_norm": round(float(grad_norm.detach().cpu()), 6),
                })
            except Exception as exc:
                record.update({
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                result["optimizer_updates"].append(record)
                break
            record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            record["gpu_peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 2
            )
            result["optimizer_updates"].append(record)
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)

        if result["optimizer_updates"] and any(
            item.get("status") == "success" for item in result["optimizer_updates"]
        ):
            model.save_pretrained(adapter_dir)
            result["adapter_saved"] = True
        else:
            result["adapter_saved"] = False
        del optimizer, model
        torch.cuda.empty_cache()
        if not result["adapter_saved"]:
            result["run_status"] = "training_failed_before_adapter_save"
            return 3

        load_start = time.perf_counter()
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": 0},
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            base, adapter_dir, is_trainable=False, local_files_only=True
        )
        result["adapter_reload_seconds"] = round(time.perf_counter() - load_start, 3)
        model.eval()
        for sample in val_samples:
            inference = infer_one(model, processor, sample, "v1_after_adapter_reload")
            result["validation_results"].append(inference)
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(inference, ensure_ascii=False), flush=True)
            if inference["status"] != "success":
                break
        result["run_status"] = (
            "success"
            if len(result["optimizer_updates"]) == STEPS
            and all(x.get("status") == "success" for x in result["optimizer_updates"])
            and len(result["validation_results"]) == len(val_samples)
            and all(x.get("status") == "success" for x in result["validation_results"])
            else "partial_or_failed"
        )
    except Exception as exc:
        result.update({
            "run_status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        result["gpu_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 2
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2

    result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"run_status": result["run_status"], "updates": len(result["optimizer_updates"]), "result_path": str(result_path)}, ensure_ascii=False))
    return 0 if result["run_status"] == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())

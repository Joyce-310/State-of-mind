# -*- coding: utf-8 -*-
"""Qwen2.5-0.5B-Instruct + LoRA fine-tuning for six Chinese emotions on Mac."""
import json, os, random
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = "data/train.jsonl"
OUTPUT_DIR = "outputs/qwen-emotion-lora"
TEST_FILE = "data/test.jsonl"
MAX_LENGTH = 192
SEED = 42
LABELS = ["開心", "悲傷", "憤怒", "恐懼", "平靜", "焦慮"]

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_rows(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"運算裝置：{device}")
    if device == "cpu":
        print("沒有偵測到 MPS，將使用 CPU，訓練會比較慢。")

    rows = load_rows(DATA_FILE)
    train_rows, test_rows = train_test_split(
        rows, test_size=0.2, random_state=SEED,
        stratify=[r["label"] for r in rows]
    )
    save_rows(test_rows, TEST_FILE)
    print(f"訓練資料：{len(train_rows)} 筆；測試資料：{len(test_rows)} 筆")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先用 float32 以提高 Apple MPS 相容性；0.5B 模型適合做入門實驗。
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def encode(row):
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text, add_special_tokens=False,
            truncation=True, max_length=MAX_LENGTH
        )["input_ids"]
        labels = full_ids.copy()
        mask_len = min(len(prompt_ids), len(labels))
        labels[:mask_len] = [-100] * mask_len
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels
        }

    train_ds = Dataset.from_list([encode(r) for r in train_rows])
    eval_ds = Dataset.from_list([encode(r) for r in test_rows])

    def collate(features):
        max_len = max(len(x["input_ids"]) for x in features)
        ids, masks, labs = [], [], []
        for x in features:
            pad = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labs.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long)
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=SEED
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=eval_ds, data_collator=collate
    )
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 獨立測試集生成分類結果
    model.eval()
    y_true, y_pred = [], []
    for row in test_rows:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        pred = next((label for label in LABELS if label in answer), "無法判斷")
        y_true.append(row["label"])
        y_pred.append(pred)

    print("\n===== 獨立測試集結果（初步）=====")
    print("Accuracy：", f"{accuracy_score(y_true, y_pred):.2%}")
    print(classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0
    ))
    print("混淆矩陣（列=真實；欄=預測）：")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))
    print("\nLoRA adapter 已儲存至：", OUTPUT_DIR)
    print("提醒：每類目前只有 20 筆示範句，測試分數不代表實際應用表現。")

if __name__ == "__main__":
    main()

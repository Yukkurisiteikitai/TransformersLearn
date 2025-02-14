import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # ← 追加
import TrainingConfig


def learn(data:TrainingConfig.TrainingConfig):
    # モデル名
    model_name = data.model_name
    training_data_json_path = data.training_data_json_path
    save_path = data.save_path
    result_path = data.Result_path
 

    # 4bit 量子化の設定 (QLoRA 用)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # トークナイザーとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",
    )

    # ここでモデルを量子化トレーニング用に準備する
    model = prepare_model_for_kbit_training(model)

    # LoRA の設定
    lora_config = LoraConfig(
        r=16,  # LoRA のランク
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # LoRA を適用する層（モデルに合わせて調整）
        bias="none",
        task_type="CAUSAL_LM",
    )

    # LoRA を適用（これ以降、LoRA 層のみが学習対象となる）
    model = get_peft_model(model, lora_config)

    # メモリ削減のための gradient checkpointing を有効化
    model.gradient_checkpointing_enable()

    # 入力埋め込みなどに対して勾配計算を有効にする（必要に応じて）
    model.enable_input_require_grads()

    # パディングトークンを EOS に設定
    tokenizer.pad_token = tokenizer.eos_token


    # チャット形式のメッセージリストを1つのプロンプト文字列に変換する関数
    def format_messages(messages):
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += "[SYSTEM] " + content + "\n"
            elif role == "user":
                prompt += "[USER] " + content + "\n"
            elif role == "assistant":
                prompt += "[ASSISTANT] " + content + "\n"
        return prompt

    # データセットの前処理関数
    def preprocess_function(examples):
        inputs = [format_messages(messages) for messages in examples["messages"]]
        tokenized_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs


    # データセットのロード
    dataset = load_dataset(
        "json", 
        data_files={"train": training_data_json_path}, 
        split="train")


    # 前処理の適用
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # DataCollator（Causal LM 用：mlm=False）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # TrainingArguments の設定
    training_args = TrainingArguments(
        output_dir=result_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,  # fp16 のほうが安定する場合があるので利用
    )


    # Trainer の初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # ファインチューニングの実行
    trainer.train()


    # Fine-tuned モデルの保存
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

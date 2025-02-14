# 学習方法の概要

このコードは、**QLoRA (Quantized Low-Rank Adaptation)** を使用して、事前学習済みの因果言語モデル (Causal Language Model) を効率的にファインチューニングする方法を実装しています。以下に、学習方法の詳細を説明します。

---

## 使用技術とライブラリ

- **Hugging Face Transformers**: モデルとトークナイザーのロード、トレーニングをサポート。
- **Datasets**: データセットのロードと前処理をサポート。
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA (Low-Rank Adaptation) を使用した効率的なファインチューニングを実現。
- **BitsAndBytes**: 4bit量子化をサポートし、メモリ効率を向上。
- **PyTorch**: モデルのトレーニングを実行。cuda version12.4を使用しています


## 学習方法の詳細

### 1. **モデルの準備**

1. **4bit量子化の設定**:
   - `BitsAndBytesConfig` を使用して、モデルを4bit量子化します。
   - メモリ使用量を削減しつつ、トレーニング可能な状態を維持します。

   ```python
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_use_double_quant=True,
   )
   ```

2. **モデルとトークナイザーのロード**:
   - Hugging Faceの`AutoTokenizer`と`AutoModelForCausalLM`を使用して、事前学習済みモデルをロードします。

   ```python
   tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       quantization_config=quantization_config,
       device_map="cuda:0",
   )
   ```

3. **量子化トレーニングの準備**:
   - `prepare_model_for_kbit_training` を使用して、量子化されたモデルをトレーニング可能な状態にします。

   ```python
   model = prepare_model_for_kbit_training(model)
   ```

4. **LoRA (Low-Rank Adaptation) の適用**:
   - LoRAを使用して、モデルの特定の層 (`q_proj`, `v_proj`) に対して効率的なファインチューニングを行います。
   - LoRAは、モデル全体ではなく一部のパラメータのみを更新することで、計算コストを削減します。

   ```python
   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       lora_dropout=0.05,
       target_modules=["q_proj", "v_proj"],
       bias="none",
       task_type="CAUSAL_LM",
   )
   model = get_peft_model(model, lora_config)
   ```

5. **追加の設定**:
   - 勾配チェックポイントを有効化してメモリ使用量を削減。
   - 入力埋め込みに対して勾配計算を有効化。
   - パディングトークンをEOSトークンに設定。

---

### 2. **データセットの準備**

1. **データセットのロード**:
   - JSON形式のデータセットを`datasets`ライブラリを使用してロードします。

   ```python
   dataset = load_dataset(
       "json", 
       data_files={"train": training_data_json_path}, 
       split="train"
   )
   ```

2. **データの前処理**:
   - チャット形式のデータを1つのプロンプト文字列に変換。
   - トークナイザーを使用してトークン化し、ラベルを設定。

   ```python
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
   ```

3. **データセットのトークン化**:
   - 前処理関数を適用して、トークン化されたデータセットを作成。

   ```python
   tokenized_dataset = dataset.map(
       preprocess_function,
       batched=True,
       remove_columns=dataset.column_names,
   )
   ```

---

### 3. **トレーニングの設定**

1. **DataCollatorの設定**:
   - `DataCollatorForLanguageModeling` を使用して、Causal Language Modeling (CLM) 用のデータコラレーターを作成。

   ```python
   data_collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer, mlm=False,
   )
   ```

2. **TrainingArgumentsの設定**:
   - トレーニングのパラメータを`TrainingArguments`で設定。
   - 例: エポック数、バッチサイズ、学習率、勾配の累積ステップ数など。

   ```python
   training_args = TrainingArguments(
       output_dir=result_path,
       num_train_epochs=3,
       per_device_train_batch_size=1,
       gradient_accumulation_steps=16,
       learning_rate=2e-5,
       logging_steps=50,
       save_steps=500,
       save_total_limit=2,
       fp16=True,
   )
   ```

3. **Trainerの初期化**:
   - モデル、トレーニングパラメータ、データセット、データコラレーターを指定して`Trainer`を初期化。

   ```python
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_dataset,
       data_collator=data_collator,
   )
   ```

---

### 4. **トレーニングの実行**

- `trainer.train()` を呼び出して、ファインチューニングを実行。

```python
trainer.train()
```

---

### 5. **モデルの保存**

- ファインチューニング後のモデルとトークナイザーを指定したディレクトリに保存。

```python
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```


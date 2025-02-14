# 概要
このコードは、QLoRA (Quantized Low-Rank Adaptation) を使用して、事前学習済みの因果言語モデル (Causal Language Model) を効率的にファインチューニングする方法を実装しています。以下に、学習方法の詳細を説明します。
[詳細な仕組み](https://github.com/Yukkurisiteikitai/TransformersLearn/blob/main/code.md)



# インストール方法

```
pip git+https://github.com/Yukkurisiteikitai/TransformersLearn.git
```


## 使い方
```python
from learn import training
from learn import TrainingConfig

class traningConfig(TrainingConfig.TrainingConfig):
    def __init__(self):
        self.model_name = "elyza/Llama-3-ELYZA-JP-8B"
        self.training_data_json_path = "learnData.json"
        self.save_path = "./save_path"
        self.Result_path = "./result"



training.learn(traningConfig)
```

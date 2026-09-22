---
language:
- en
pipeline_tag: text-classification
license: mit
---

# MiniCheck-Flan-T5-Large

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s-5TYnGV3kGFMLp798r5N-FXPD8lt2dm?usp=sharing)

This is a fact-checking model from our work:

📃 [**MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents**](https://arxiv.org/pdf/2404.10774.pdf),(EMNLP 2024, [GitHub Repo](https://github.com/Liyan06/MiniCheck))

The model is based on Flan-T5-Large that predicts a binary label - 1 for supported and 0 for unsupported. 
The model is doing predictions on the *sentence-level*. It takes as input a document and a sentence and determine 
whether the sentence is supported by the document: **MiniCheck-Model(document, claim) -> {0, 1}**


**MiniCheck-Flan-T5-Large is the best fack-checking model with size < 1B** and reaches GPT-4 performance. It is fine tuned from `google/flan-t5-large` ([Chung et al., 2022](https://arxiv.org/pdf/2210.11416.pdf)) 
on the combination of 35K data:
- 21K ANLI data ([Nie et al., 2020](https://aclanthology.org/2020.acl-main.441.pdf))
- 14K synthetic data generated from scratch in a structed way (more details in the paper).


### Model Variants
We also have other three MiniCheck model variants:
- [bespokelabs/Bespoke-Minicheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) (Model Size: 7B)
- [lytang/MiniCheck-RoBERTa-Large](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) (Model Size: 0.4B)
- [lytang/MiniCheck-DeBERTa-v3-Large](https://huggingface.co/lytang/MiniCheck-DeBERTa-v3-Large) (Model Size: 0.4B)


### Model Performance

<p align="center">
    <img src="./performance.png" width="550">
</p>



The performance of these models is evaluated on our new collected benchmark (unseen by our models during training), [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact), 
from 11 recent human annotated datasets on fact-checking and grounding LLM generations. MiniCheck-Flan-T5-Large outperform all
exisiting specialized fact-checkers with a similar scale by a large margin (4-10% absolute increase) and is on par with GPT-4, but 400x cheaper. See full results in our work.

Note: We only evaluated the performance of our models on real claims -- without any human intervention in 
any format, such as injecting certain error types into model-generated claims. Those edited claims do not reflect
LLMs' actual behaviors.


# Model Usage Demo

Please run the following command to install the **MiniCheck package** and all necessary dependencies.
```sh
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
```

### Below is a simple use case

```python
from minicheck.minicheck import MiniCheck
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

doc = "A group of students gather in the school library to study for their upcoming final exams."
claim_1 = "The students are preparing for an examination."
claim_2 = "The students are on vacation."

# model_name can be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']
scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

print(pred_label) # [1, 0]
print(raw_prob)   # [0.9805923700332642, 0.007121307775378227]
```

### Test on our [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) Benchmark

```python
import pandas as pd
from datasets import load_dataset
from minicheck.minicheck import MiniCheck
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load 29K test data
df = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")['test'])
docs = df.doc.values
claims = df.claim.values

scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)  # ~ 500 docs/min, depending on hardware
```

To evalaute the result on the benchmark
```python
from sklearn.metrics import balanced_accuracy_score

df['preds'] = pred_label
result_df = pd.DataFrame(columns=['Dataset', 'BAcc'])
for dataset in df.dataset.unique():
    sub_df = df[df.dataset == dataset]
    bacc = balanced_accuracy_score(sub_df.label, sub_df.preds) * 100
    result_df.loc[len(result_df)] = [dataset, bacc]

result_df.loc[len(result_df)] = ['Average', result_df.BAcc.mean()]
result_df.round(1)
```

# Citation

```
@InProceedings{tang-etal-2024-minicheck,
  title = {MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents},
  author = {Liyan Tang and Philippe Laban and Greg Durrett},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year = {2024},
  publisher = {Association for Computational Linguistics},
  url = {https://arxiv.org/pdf/2404.10774}
}
```
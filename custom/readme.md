## Use local API server

Set this in the VLMEVALKIT/.env file.

```
LOCAL_LLM=local_server
OPENAI_API_BASE=http://localhost:8000/v1
```

The command line argument should be set as "--judge chatgpt-1106" even when a local API server is used. That is because the hardcoded  requirement in different datasets.

## Model environment setup

Don't need install from github: MiniCPM, Phi3, Qwen2-VL
Need install from the original github repo: VILA, LLava-Next
For Qwen2.5_VL, min transformers: 4.49.0.dev0

The supported datasets are defiend in `vlmeval/dataset/__init__.py`.

## Datasets

When modifying or adding datasets, to avoid merging conflicts, do not change any existing lines. Just add new lines.

**Modify existing dataset**

* If needed, define a new dataset class in `vlmeval/dataset/`. See e.g. `vlmeval/dataset/tempcompass_MCQ_YorN.py`.
* Register this dataset tyep in `vlmeval/dataset/__init__.py`. For example,
```python
from .tempcompass_MCQ_YorN import TempCompass_MCQ_YorN
```

```python
CUSTOM_DATASET = [        
    TempCompass_MCQ_YorN
```
* If needed, if this dataset needs pre-configured input parameters, define a partial in `vlmeval/dataset/video_dataset_config.py`, such as
```python
tempcompass_dataset = {    
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_MCQ_YorN_8frame': partial(TempCompassMCQ_YorN, dataset='TempCompass', nframe=8),
```

**Define a new dataset**

* If needed, define a new dataset class in `vlmeval/dataset/`. See e.g. `vlmeval/dataset/tempcompass_MCQ_YorN.py`.
* Register this dataset tyep in `vlmeval/dataset/__init__.py`. For example,
```python
CUSTOM_DATASET = [    
    Virat_MCQ
```

The json files like "multi-choice_backup.json" are not used. Only tsv files are used.
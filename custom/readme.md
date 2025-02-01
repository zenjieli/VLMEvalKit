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

The json files like "multi-choice_backup.json" are not used. Only tsv files are used.
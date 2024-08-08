# Setup

Don't need install from github: MiniCPM, Phi3
Need install from the original github repo: VILA, LLava

## Models
### MiniCPM

Add a symbolic link to the MiniCPM weights.

```shell
ln -s ~/weights/hf/Qwen1.5-14B-Chat-GPTQ-Int4 openbmb/MiniCPM-Llama3-V-2_5-int4
```

## Datasets

The supported datasets are defiend in `vlmeval/dataset/__init__.py`.
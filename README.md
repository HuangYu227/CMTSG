# CMTSG

CMTSG v1 implements multimodal causal text conditioning, data-driven
environment routing, and a DiT-style diffusion generator for multivariate time
series.

## Expected Data Layout

Put VerbalTS-style datasets under:

```text
E:\Research\TSG\CMTSG\datasets\weather
E:\Research\TSG\CMTSG\datasets\synth-m
```

Each dataset directory should contain:

```text
meta.json
train_ts.npy
train_text_caps.npy
valid_ts.npy
valid_text_caps.npy
test_ts.npy
test_text_caps.npy
```

Pretrained models are expected at:

```text
E:\Research\TSG\CMTSG\pretrained\Qwen2.5-VL-7B-Instruct
E:\Research\TSG\CMTSG\pretrained\LongCLIP
```

## Offline Preprocessing

To copy the Weather files currently stored on the desktop into the expected
project layout:

```powershell
python -m cmtsg.prepare_data --dataset weather --source "C:\Users\蜂窝煤\Desktop\npy\weather datasets"
```

Render line charts and extract causal text with Qwen2.5-VL:

```powershell
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split train
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split valid
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split test
```

For a dry smoke run without loading Qwen:

```powershell
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split train --limit 2 --mock
```

Encode `generation_condition` with frozen LongCLIP into 64-dimensional cached
embeddings:

```powershell
python -m cmtsg.preprocess.encode_longclip --dataset weather --split train
python -m cmtsg.preprocess.encode_longclip --dataset weather --split valid
python -m cmtsg.preprocess.encode_longclip --dataset weather --split test
```

## Train

```powershell
python -m cmtsg.train --config configs/weather.yaml --epochs 10
python -m cmtsg.train --config configs/synth_m.yaml --epochs 10
```

Training reads only cached `processed/{dataset}/{split}_text_emb.npy`; Qwen and
LongCLIP are not loaded by the training loop.

## Quick Checks

```powershell
python -m compileall cmtsg tests
python tests/test_imaging.py
python tests/test_shapes.py
```

`tests/test_shapes.py` requires PyTorch.

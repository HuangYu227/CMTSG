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
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split train --caption-policy first --batch-size 4
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split valid --caption-policy first --batch-size 4
python -m cmtsg.preprocess.qwen_causal_text --dataset weather --split test --caption-policy first --batch-size 4
```

`--caption-policy first` is the default practical preprocessing mode. Use
`--caption-policy all` only when you explicitly want all VerbalTS captions,
because Weather has three captions per sample and this triples Qwen inference.

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

Training writes trend logs to:

```text
runs/{dataset}/logs/epoch_metrics.csv
runs/{dataset}/logs/epoch_metrics.jsonl
runs/{dataset}/logs/sample_metrics.jsonl
```

Per-epoch logs include train/validation diffusion loss and routing statistics.
Every `evaluation.sample_every` epochs, validation samples are generated and
scored with `MDD`, `flat_kl`, `MMD-RBF`, `fid_raw_proxy`, and
`jftsd_text_proxy`. These two proxy metrics are not VerbalTS metrics; they are
only lightweight diagnostics.

With the default configs, CTTP metrics are required. If `Weather_cttp` or
`synth-m_cttp` is missing or malformed, evaluation raises immediately and writes
`metrics_failed.json`. When CTTP is available, the sampler reports the VerbalTS
style metrics:

```text
cttp
fid_cttp
jftsd_cttp
```

These training-time CTTP numbers are for internal model selection. They are
computed on the sampled validation subset and are not protocol-identical to the
official VerbalTS evaluator.

For a paper-style comparison with VerbalTS, evaluate a saved checkpoint with the
full VerbalTS protocol:

```bash
python -m cmtsg.evaluate_verbalts_protocol \
  --config configs/synth_m.yaml \
  --checkpoint runs/synth-m/checkpoints/best_jftsd_cttp.pt \
  --output-root runs/synth-m \
  --verbalts-root ../VerbalTS \
  --cttp-root pretrained/synth-m_cttp \
  --split test \
  --sampler ddim \
  --n-samples 10 \
  --metric-caption-source original
```

This script matches the important VerbalTS evaluation choices: full test split,
10 generated samples per condition with median aggregation, train-split CTTP
feature caches for FID/JFTSD, and the VerbalTS Frechet-distance implementation.

Use `evaluation.require_cttp: false` only for debugging infrastructure without
semantic metrics.

Weather ablation configs:

```text
configs/weather.yaml                         # full CMTSG: text + GAF env + text routing
configs/ablations/weather_no_env.yaml        # removes environment conditioning
configs/ablations/weather_no_text.yaml       # removes text conditioning, routes env by learned query
configs/ablations/weather_learned_env.yaml   # replaces GAF environments with learned random env bank
configs/ablations/weather_uniform_route.yaml # replaces text routing with uniform env mixture
```

Optional CTTP links on the server:

```bash
ln -sfn /home/newuser001/huangyu/Research/VerbalTS/save/Weather_cttp pretrained/Weather_cttp
ln -sfn /home/newuser001/huangyu/Research/VerbalTS/save/synth-m_cttp pretrained/synth-m_cttp
```

Validate CTTP before a long run:

```bash
python -m cmtsg.validate_cttp --verbalts-root ../VerbalTS --cttp-root pretrained/Weather_cttp
python -m cmtsg.validate_cttp --verbalts-root ../VerbalTS --cttp-root pretrained/synth-m_cttp
```

## Quick Checks

```powershell
python -m compileall cmtsg tests
python tests/test_imaging.py
python tests/test_shapes.py
```

`tests/test_shapes.py` requires PyTorch.

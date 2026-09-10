<div align="center">

# TherA: Thermal-Aware Visual-Language Prompting for<br>Controllable RGB-to-Thermal Infrared Translation

[**Dong-Guw Lee**](https://scholar.google.com/citations?user=u6VDnlgAAAAJ&hl=ko)<sup>1*</sup>&emsp;
[**Tai Hyoung Rhee**](https://scholar.google.com/citations?user=PF8EfdYAAAAJ&hl=en&oi=ao)<sup>1*</sup>&emsp;
[**Hyunsoo Jang**](https://rpm.snu.ac.kr)<sup>1</sup><br>
[**Young-Sik Shin**](https://scholar.google.com/citations?user=gGfBRawAAAAJ&hl=en&oi=ao)<sup>2</sup>&emsp;
[**Ukcheol Shin**](https://scholar.google.com/citations?user=ZvxI80EAAAAJ&hl=ko&oi=ao)<sup>3</sup>&emsp;
[**Ayoung Kim**](https://ayoungk.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>Seoul National University&emsp;
<sup>2</sup>Kyungpook National University&emsp;
<sup>3</sup>KENTECH
<sup>*</sup> Equal Contribution&emsp;
<sup>&dagger;</sup> Corresponding Author

**CVPR 2026**

[![Project Page](https://img.shields.io/badge/Project_Page-TherA-blue)](https://donkeymouse.github.io/thera_cvpr26/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.19430-b31b1b.svg)](https://arxiv.org/abs/2602.19430)
[![GitHub](https://img.shields.io/badge/GitHub-donkeymouse%2FTherA-black)](https://github.com/donkeymouse/TherA)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-yellow)](https://huggingface.co/donkeymouse/TherA/tree/main)
[![Dataset](https://img.shields.io/badge/HuggingFace-R2T2-orange)](https://huggingface.co/datasets/donkeymouse/TherA-R2T2)
[![Docker](https://img.shields.io/badge/Docker-donkeymouse%2Fthera-2496ED)](https://hub.docker.com/r/donkeymouse/thera)

</div>

<p align="center">
  <img src="assets/method.png" width="90%" alt="TherA method overview">
</p>

> **Fork note** — this is a fork of [donkeymouse/TherA](https://github.com/donkeymouse/TherA)
> with fixes that make on-the-fly TherA-VLM inference run. See
> [Differences from Upstream](#differences-from-upstream).


---

## News

- **2026-04-03**: TherA github repo opening
- **2026-05-22**: TherA inference code and R2T2 dataset release.

---

## Overview

**TherA** is a controllable RGB-to-thermal infrared translation framework. Given an RGB image, TherA synthesizes a long-wave thermal infrared image using a latent-diffusion translator conditioned on thermal-aware visual-language features.

TherA is designed for:

- **RGB → TIR translation** for thermal perception research.
- **Thermal-aware VLM conditioning** using LLaVA hidden-state features.
- **Scene- and object-level controllability** across weather, time of day, and object state.
- **Reference-cache inference**, allowing deployment without loading LLaVA at runtime.


<div align="center">
  <a href="https://www.youtube.com/watch?v=X60UxjGKQkg">
    <img src="https://img.youtube.com/vi/X60UxjGKQkg/0.jpg" alt="TherA demo video" width="720">
  </a>
</div>


---

## Key Idea

TherA does **not** condition directly on raw text during diffusion inference. Instead, it uses a **4096-dimensional LLaVA hidden state**, either:

1. loaded from a precomputed `.pt` reference cache, or
2. extracted on the fly using LLaVA.



For resource limited environments, we recommend **reference-cache mode**. This mode uses precomputed LLaVA features such as `SUNNY.pt`, `CLOUDY.pt`, `RAINY.pt`, or `NIGHT.pt`, and therefore does **not** require loading LLaVA weights at runtime. An alternative would be to compute pre-computed LLaVA feature first followed by inferencing with reference-cache mode (upcoming feature).

---
## Repository Layout

```text
TherA/
├── infer_custom.py             # Batch RGB → TIR inference on a folder
├── infer_example_guided.py     # Single-image / example-guided inference
├── infer_palette.sh        # Run multiple weather/style palettes
├── lavi_ip2p/                  # UNet 8-channel + adapter wrapper
├── LaVi-Bridge/modules/        # TextAdapter architecture
├── llava/                      # LLaVA code, only needed for on-the-fly mode
├── thera_paths.py              # Default local weight paths
├── thera_llava.py              # Lazy LLaVA loader
└── weights/                    # Download weights here; not tracked by git
    ├── model.pt                # TherA UNet + adapter checkpoint
    ├── merged_models/          # Initialization model
    │   ├── unet/
    │   └── adapter/
    ├── stable-diffusion/
    │   ├── vae/
    │   └── scheduler/
    ├── palettes/               # Precomputed LLaVA features (reference-cache mode)
    │   ├── SUNNY.pt
    │   ├── CLOUDY.pt
    │   ├── RAINY.pt
    │   └── NIGHT.pt
    ├── TherA_VLM/              # Optional; LoRA adapter for on-the-fly mode
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── config.json
    │   ├── non_lora_trainables.bin
    │   └── trainer_state.json
    └── llava-v1.5-7b/          # Optional; LLaVA base for on-the-fly mode
```

---

## Installation

### Option 1: Local Python Environment

```bash
git clone https://github.com/AhaanKotian/TherA.git
cd TherA

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Recommended environment**

- Python 3.10+
- CUDA-capable GPU
- 16 GB+ VRAM recommended for comfortable inference

---

### Option 2: Docker

A prebuilt Docker image is available at:

```bash
docker pull donkeymouse/thera:latest
```

Example interactive run:

```bash
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace/TherA \
  -w /workspace/TherA \
  donkeymouse/thera:latest \
  bash
```

Then run inference commands from inside the container.

---

## Download Weights

TherA weights are hosted on Hugging Face:

```bash
pip install -U huggingface_hub

huggingface-cli download donkeymouse/TherA \
  --local-dir weights
```

After downloading, your `weights/` directory should contain:

| Path | Description | Required? |
|---|---|---|
| `weights/model.pt` | TherA trained UNet and adapter checkpoint | Yes |
| `weights/merged_models/unet/` | UNet architecture/config files | Yes |
| `weights/merged_models/adapter/` | TextAdapter architecture/config files | Yes |
| `weights/stable-diffusion/vae/` | Stable Diffusion VAE | Yes |
| `weights/stable-diffusion/scheduler/` | DDIM scheduler config | Yes |
| `weights/palettes/*.pt` | Precomputed LLaVA hidden states for inference palettes | Recommended |
| `weights/TherA_VLM/` | LLaVA LoRA adapter for on-the-fly feature extraction | Optional |

> **Folder names.** The download gives you `palettes/` and `TherA_VLM/` (underscore).
> Earlier revisions of this README called these `reference_caches/` and `TherA-VLM/`;
> the commands below use the names you actually get. Note that `infer_palette.sh` still
> hardcodes `weights/reference_caches/`, so rename the folder or edit that path before
> running it.

---

## Optional: Download LLaVA Weights

LLaVA is only required for **on-the-fly feature extraction** or **two-image guided mode**. It is not required for reference-cache inference.

TherA-VLM is a LoRA adapter over **original-format** LLaVA v1.5-7B, so the base model must be
`liuhaotian/llava-v1.5-7b` (~14 GB):

```bash
huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir weights/llava-v1.5-7b
```

> **Do not use `llava-hf/llava-1.5-7b-hf`.** That is the HF-native port
> (`LlavaForConditionalGeneration`, with nested `vision_config`/`text_config`). The `llava/`
> package bundled here is the original haotian-liu implementation, which builds its vision
> tower from a top-level `mm_vision_tower` config key. With the HF port that key is absent, so
> no vision tower is constructed and loading dies with
> `AttributeError: 'NoneType' object has no attribute 'is_loaded'`. The weight names would not
> have matched either — the 7B language model would have loaded as **random weights, silently**.
> `weights/TherA_VLM/config.json` declares `LlavaLlamaForCausalLM` with
> `mm_vision_tower: openai/clip-vit-large-patch14-336`, which matches `liuhaotian/llava-v1.5-7b`
> field-for-field. (Upstream's instruction appears to stem from the authors' local folder simply
> being *named* `llava-1.5-7b-hf`.)

The CLIP vision tower is **not** part of that repo. It is fetched on first load from
`openai/clip-vit-large-patch14-336` (~1.7 GB, cached under `~/.cache/huggingface`). Pre-download
it if the machine will be offline at inference time:

```bash
huggingface-cli download openai/clip-vit-large-patch14-336
```

**Keep `llava` in the directory name.** `load_pretrained_model` dispatches on the last path
segment rather than on the config, so a base directory without `llava` in its name silently
falls through to a plain `AutoModelForCausalLM` load — no vision tower, `image_processor=None`.
Avoid `lora` and `mistral` in the name too; they select other branches.


---

## Quick Start

The repo ships three sample RGB images in `assets/`. Stage an input folder:

```bash
mkdir -p examples/rgb
cp assets/testA.jpg assets/testB.png assets/testC.png examples/rgb/
```

Then run reference-cache mode, which needs no LLaVA weights:

```bash
python infer_custom.py \
  --rgb-dir examples/rgb \
  --output-dir preds/sunny \
  --reference-cache weights/palettes/SUNNY.pt
```

---

## Full RGB-TIR translation using TherA-VLM

Use this mode if you want to extract hidden states from TherA directly at runtime from an RGB image and prompt.

```bash
python infer_custom.py \
  --rgb-dir examples/rgb \
  --output-dir preds \
  --llava-base-path weights/llava-v1.5-7b \
  --llava-lora-path weights/TherA_VLM \
  --llava-prompt "How would this RGB scene appear in long-wave thermal infrared spectrum."
```

This mode is more expensive because it loads LLaVA during inference. On startup the extractor
validates the base checkpoint and fails fast with an actionable message if it is the wrong
LLaVA format.



### Reference-Guided Image Translation Mode

This mode extracts LLaVA features from a reference RGB image and applies them to a target RGB image.

```bash
python infer_example_guided.py \
  --mode two-image \
  --reference-image examples/ref/rgb.jpg \
  --input-image examples/rgb/scene.jpg \
  --output preds/scene_tir.png \
  --llava-base-path weights/llava-v1.5-7b \
  --llava-lora-path weights/TherA_VLM
```

---

### Recursive Folder Inference

```bash
python infer_custom.py \
  --rgb-dir /path/to/dataset/RGB \
  --output-dir preds \
  --reference-cache weights/palettes/SUNNY.pt \
  --recursive
```

When `--recursive` is used, the output folder preserves the input directory structure.

---
---

## Reference-cache Mode
Reference-cache mode is the recommended if you are lacking GPU memory. It does not load LLaVA at runtime.

```bash
python infer_custom.py \
  --rgb-dir examples/rgb \
  --output-dir preds/sunny \
  --reference-cache weights/palettes/SUNNY.pt
```

The script reads all images in `examples/rgb` and writes translated TIR images to `preds/sunny`.

A lighter version of the text-guided image translation module.

Example palette caches:

```text
weights/palettes/SUNNY.pt
weights/palettes/CLOUDY.pt
weights/palettes/RAINY.pt
weights/palettes/NIGHT.pt
```

You can use different pallete cache to achieve different translation effects.

---

## Inference Modes

| Mode | Main flag / script | LLaVA weights needed? | Recommended use |
|---|---|---:|---|
| Reference cache | `--reference-cache path.pt` | No | Default deployment and fast inference |
| Per-image cache directory | `--cache-dir dir/` | No | Precomputed feature per image |
| Full RGB-TIR translation| `--llava-base-path ...` | Yes | Runtime prompt/image conditioning |
| Reference image-guided translation| `infer_example_guided.py --mode two-image` | Yes | Apply reference-image conditioning |

---

## Reference Cache Format

Reference caches are precomputed LLaVA hidden states saved as `.pt` files.

Supported tensor shapes:

```text
[1, L, 4096]
[L, 4096]
```

A single reference cache can be applied to all input images as a global thermal/weather/style condition.

---

## Important Arguments

| Argument | Default | Description |
|---|---:|---|
| `--checkpoint` | `weights/` | Directory containing `model.pt` |
| `--merged-model-path` | `weights/merged_models` | Directory containing UNet and adapter configs |
| `--pretrained-sd` | `weights/stable-diffusion` | Directory containing VAE and scheduler |
| `--rgb-dir` | Required | Folder of RGB images for batch inference |
| `--output-dir` | `custom_predictions` | Output folder for predictions |
| `--reference-cache` | `None` | Single `.pt` cache used for all images |
| `--cache-dir` | `None` | Folder of per-image `.pt` caches matched by filename stem |
| `--llava-base-path` | `None` | Base LLaVA model path for on-the-fly mode |
| `--llava-lora-path` | `None` | Optional LLaVA LoRA path |
| `--llava-prompt` | thermal prompt | Prompt used for default inference/text-guided translation|
| `--num-steps` | `100` | DDIM sampling steps |
| `--cfg-text` | `3.5` | Text/VLM guidance strength |
| `--cfg-image` | `1.5` | Image guidance strength |
| `--target-size` | Auto | Resize image to this square size; otherwise dimensions are rounded to multiples of 32 |
| `--recursive` | Off | Recursively process subdirectories |
| `--device` | `cuda` | Device for inference |

---

## Architecture

```text
RGB image
   │
   ▼
VAE encoder ──► RGB latents ───────────────────┐
                                                │
                                                ├──► 8-channel diffusion UNet ──► VAE decoder ──► TIR image
                                                │
LLaVA hidden state, 4096-d ──► TextAdapter ─────┘
                              768-d cross-attention tokens
```

TherA uses dual classifier-free guidance at inference by combining:

- full conditioning,
- image-only conditioning,
- text/VLM-only conditioning.

---

## R2T2 Dataset

TherA is trained with **R2T2**, a large-scale RGB–TIR–Text dataset.

R2T2 includes:

- **112,970 aligned triplets**: RGB image, TIR image, and canonical thermal schema.
- Scene diversity across driving, CCTV, aerial, and ego-view settings.
- Temporal diversity across day/night and diurnal transitions.
- Environmental diversity across weather, season, and illumination.
- Material- and object-level annotations with structured canonicalization.
- Data compiled from multiple aligned RGB–TIR datasets with additional pseudo-aligned pairs.

Dataset page:

```text
https://huggingface.co/datasets/donkeymouse/TherA-R2T2
```

Example structure:

```text
R2T2/
├── ${DATASET_NAME}/
│   └── ${SEQUENCE_NAME}/
│       ├── RGB/
│       │   ├── 1.jpg
│       │   └── ...
│       └── TIR/
│           ├── 1.jpg
│           └── ...
├── ViVID/
│   ├── img_campus_day1/
│   │   ├── RGB/
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── TIR/
│   │       ├── 000001.png
│   │       └── ...
│   └── ...
└── ...
```

---

## Troubleshooting

### `Checkpoint not found: weights/model.pt`

Download the TherA weights and make sure `model.pt` is located at:

```text
weights/model.pt
```

---

### `OSError: ... stable-diffusion/vae`

Make sure the Stable Diffusion VAE and scheduler folders are present:

```text
weights/stable-diffusion/vae/
weights/stable-diffusion/scheduler/
```

---

### Outputs look identical across palettes

Try increasing text/VLM guidance:

```bash
--cfg-text 7.5
```

Also verify that your reference cache files are distinct:

```text
SUNNY.pt
CLOUDY.pt
RAINY.pt
NIGHT.pt
```

---

### CUDA out of memory

Try reducing the image size:

```bash
--target-size 512
```

You can also run one image at a time with:

```bash
python infer_example_guided.py
```

---

### `AttributeError: 'NoneType' object has no attribute 'is_loaded'`

The base LLaVA checkpoint is the HF-native port instead of the original format, so no vision
tower was built. Download `liuhaotian/llava-v1.5-7b` — see
[Optional: Download LLaVA Weights](#optional-download-llava-weights). This fork validates the
config up front, so the wrong checkpoint now surfaces as a `ValueError` naming the correct
download before any weights load.

---

### LLaVA import or loading errors

Use reference-cache mode if you do not need runtime LLaVA extraction:

```bash
--reference-cache weights/palettes/SUNNY.pt
```

For on-the-fly mode, make sure the LLaVA base model and TherA weights are correctly loaded.

---

## TODOs
- [x] inference code and R2T2 dataset
- [] Upload cache extraction code
- [] Improve text-guidance


## Citation

If you find TherA useful for your research, please cite:

```bibtex
@inproceedings{lee2026thera,
  title     = {TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation},
  author    = {Lee, Dong-Guw and Rhee, Tai Hyoung and Jang, Hyunsoo and Shin, Young-Sik and Shin, Ukcheol and Kim, Ayoung},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

You may also cite the arXiv version:

```bibtex
@article{lee2026thera_arxiv,
  title   = {TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation},
  author  = {Lee, Dong-Guw and Rhee, Tai Hyoung and Jang, Hyunsoo and Shin, Young-Sik and Shin, Ukcheol and Kim, Ayoung},
  journal = {arXiv preprint arXiv:2602.19430},
  year    = {2026}
}
```

---

## Differences from Upstream

This fork changes the following relative to
[donkeymouse/TherA](https://github.com/donkeymouse/TherA):

| Change | Where |
|---|---|
| Base LLaVA checkpoint corrected to `liuhaotian/llava-v1.5-7b`; upstream pointed at the incompatible HF-native `llava-hf/llava-1.5-7b-hf` | README |
| Preflight validation of the LLaVA base checkpoint — raises a `ValueError` naming the correct download instead of a `NoneType` traceback, and warns on a base directory name that would silently disable the multimodal branch | `llava/model/frozen_llava_extractor.py` |
| Docstring examples updated off the stale `llava-miragehd-*` paths | `infer_custom.py`, `infer_example_guided.py` |
| Weight paths documented as `palettes/` and `TherA_VLM/`, matching the actual Hugging Face layout | README |
| `--checkpoint` default documented as `weights/` | README |

---

## Acknowledgements

TherA builds on open-source components from the vision-language and diffusion communities, including LLaVA, Stable Diffusion, Diffusers, and LaVi-Bridge-style adapter architectures.

---

## License

See `LICENSE` for details.

Third-party models, datasets, and libraries retain their own licenses. Please review the licenses for LLaVA, Stable Diffusion, Hugging Face model files, and any external datasets before use.

## Contact
If you have any questions, contact here please
```
donkeymouse@snu.ac.kr
```

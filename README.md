# LiMA: Less is More for Attribution

Official PyTorch implementation of **[Less is More: Efficient Black-box Attribution via Minimal Interpretable Subset Selection](https://arxiv.org/abs/2504.00470)**.

[![arXiv](https://img.shields.io/badge/arXiv-2504.00470-b31b1b.svg)](https://arxiv.org/abs/2504.00470)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LiMA (**Less input is More faithful for Attribution**) formulates black-box attribution as submodular subset selection. It ranks interpretable input regions with bidirectional greedy search and requires model outputs rather than gradients. Across eight foundation models, the paper reports average improvements of 36.3% in Insertion and 39.6% in Deletion, and a 1.6x speedup over naive greedy search.

> This is research code. Several model-specific files configure paths, methods, and GPU IDs as constants; inspect them before launching a full experiment.

## Repository layout

```text
datasets/                 Evaluation lists (data is not included)
models/                   LiMA objectives and search implementations
submodular_attribution/   LiMA generation/inference entry points
scripts/                  Main experiments and ablations
evals/                    Faithfulness, MuFidelity, and error analysis
baseline_attribution/     Baseline generation and inference
tutorial/                 Model- and task-specific notebooks
visualization/            Visualization utilities
```

## Installation

We recommend Linux, Python 3.8+, and a CUDA-capable GPU.

```bash
git clone https://github.com/RuoyuChen10/LIMA.git
cd LIMA
conda create -n lima python=3.9 -y
conda activate lima

pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn scikit-image opencv-contrib-python \
  pillow matplotlib tqdm timm transformers xplique tensorflow
```

Install the upstream packages needed by the selected experiment:

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [ImageBind](https://github.com/facebookresearch/ImageBind)
- [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind)
- [Segment Anything](https://github.com/facebookresearch/segment-anything), for SAM regions

Some optional baselines additionally require packages such as `lime` or `tensorflow-addons`. A universal lock file is not provided because the upstream model stacks have different requirements.

## Data and checkpoints

### ImageNet

Download ILSVRC 2012 from the [official ImageNet website](https://www.image-net.org/) and put the flat validation set at:

```text
datasets/imagenet/ILSVRC2012_img_val/
├── ILSVRC2012_val_00000001.JPEG
└── ...
```

Evaluation lists for each backbone are included, such as `val_clip_vitl_5k_true.txt`, `val_imagebind_5k_true.txt`, and `val_languagebind_5k_true.txt`. Each line is:

```text
<relative-file-name> <class-index>
```

`*_true.txt` contains correctly classified samples used for the main faithfulness experiments. `*_false.txt` contains misclassified samples used for prediction-error analysis.

### VGGSound and LC25000

- Put VGGSound `.flac` files in `datasets/vggsound/test/`.
- Put LC25000 lung images in `datasets/medical_lung/lung_dataset/`.

The datasets are not redistributed. Obtain them under their own terms and preserve the filenames in the supplied lists.

### Model assets

Upstream models use `.checkpoints/`. The main ImageNet scripts expect zero-shot semantic weights under `ckpt/semantic_features/`, for example:

```text
clip_vitl_imagenet_zeroweights.pt
imagebind_imagenet_zeroweights.pt
languagebind_imagenet_zeroweights.pt
```

Some entry points create missing semantic features; others expect them to exist. Check the selected file before a long run.

## Run LiMA

Run commands from the repository root. Multi-GPU scripts divide the evaluation list over the IDs in `cuda_devices`; edit that array for your machine.

```bash
# Main ImageNet experiments
bash scripts/efficientv2-clip_vitl_multigpu.sh
bash scripts/efficientv2-clip_rn101_multigpu.sh
bash scripts/efficientv2-imagebind_multigpu.sh
bash scripts/efficientv2-languagebind_multigpu.sh
bash scripts/vitl.sh
```

The `efficientv2-*` scripts run the paper's efficient bidirectional search. Scripts without `efficientv2` run the earlier/plain search; `efficientv1-*` is the first efficient variant. The `ablation-*` scripts vary the objective, candidate count, or region division.

For a ten-image, single-GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel \
  --Datasets datasets/imagenet/ILSVRC2012_img_val \
  --eval-list datasets/imagenet/val_clip_vitl_5k_true.txt \
  --superpixel-algorithm seeds \
  --lambda1 0 --lambda2 0.05 --lambda3 20 --lambda4 1 \
  --pending-samples 8 --begin 0 --end 10 \
  --save-dir submodular_results/imagenet-clip-vitl-efficientv2
```

Output layout:

```text
submodular_results/.../<configuration>/
├── npy/<class-id>/<sample>.npy    # Regions in selected/ranked order
└── json/<class-id>/<sample>.json  # Scores along the perturbation path
```

The JSON includes `consistency_score` and `collaboration_score`, consumed by the faithfulness evaluator.

The `*_debug.sh` scripts use `_false.txt` splits to explain prediction errors. Despite their historical name, these are inference experiments, not debugger utilities.

## Evaluation

### LiMA Insertion and Deletion

Pass the configuration directory containing both `npy/` and `json/`:

```bash
python -m evals.eval_AUC_faithfulness \
  --explanation-dir submodular_results/imagenet-clip-vitl-efficientv2/seeds-0.0-0.05-20.0-1.0-pending-samples-8
```

The command prints mean Insertion and Deletion AUC. Higher Insertion and lower Deletion indicate better faithfulness. For audio, use `evals.eval_AUC_faithfulness_audio`.

### MuFidelity

LiMA MuFidelity evaluators read the generated `npy/` directory:

```bash
python -m evals.evaluation_mufidelity_imagenet_CLIP \
  --Datasets datasets/imagenet/ILSVRC2012_img_val \
  --eval-list datasets/imagenet/val_clip_vitl_5k_true.txt \
  --eval-number -1 \
  --explanation-smdl submodular_results/imagenet-clip-vitl-efficientv2/seeds-0.0-0.05-20.0-1.0-pending-samples-8/npy
```

Use the corresponding ImageBind or LanguageBind file for those backbones. `--eval-number` limits the evaluated subset; in scripts using Python slicing, `-1` selects all but the final empty line.

### Prediction-error analysis

After running an `efficientv2-*_multigpu_debug.sh` experiment, set `explanation_method` at the top of `evals/evaluation_mistake_debug_ours.py`, then run:

```bash
python -m evals.evaluation_mistake_debug_ours
```

It reports the highest target confidence and retained-input percentage. The CNN counterpart is `evaluation_mistake_debug_ours_cnn.py`.

## Baselines

The baseline pipeline has three stages:

1. **Generate maps:** choose a model-specific `baseline_attribution/generate_*.py` file and configure its dataset, output, GPU, and `explainers` list.
2. **Run perturbation inference:** point the matching `debug_org_attribution_method_*.py` file at those `.npy` maps. It writes score-curve JSON files under `explanation_insertion_results/`.
3. **Aggregate metrics:** run `eval_AUC_faithfulness_baseline.py`, or use a model-specific `eval_*_baseline.py` / `evaluation_mufidelity_*_baseline.py` directly on the maps.

Example for CLIP ViT-L/14:

```bash
# Edit methods/paths in each file first.
python -m baseline_attribution.generate_explanation_maps_clip_vitl
python -m baseline_attribution.debug_org_attribution_method_clip_vitl
python -m evals.eval_AUC_faithfulness_baseline \
  --explanation-dir explanation_insertion_results/imagenet-clip-vitl-true/<method>
```

The generators cover Xplique methods such as Saliency, Integrated Gradients, RISE, KernelSHAP, and HSIC, plus bundled Grad-ECLIP, IGOS++, IG2, ViT-CX, and SAMP implementations. Most are experiment scripts with constants rather than full CLI programs. Remove any sample-specific `if ...: continue` filter left in a `debug_org_*` script before evaluating a full split.

Baseline maps can also be evaluated directly:

```bash
python -m evals.eval_clip_vitl_baseline \
  --Datasets datasets/imagenet/ILSVRC2012_img_val \
  --eval-list datasets/imagenet/val_clip_vitl_5k_true.txt \
  --eval-number -1 \
  --explanation-method explanation_results/imagenet-clip-vitl-true/KernelShap
```

## Tutorials

See `tutorial/` for CLIP, ImageBind, LanguageBind, CUB, VGGSound, Quilt, and SAM examples. `visualization/` contains utilities for rendering saved explanations.

## Citation

```bibtex
@article{chen2025less,
  title   = {Less is More: Efficient Black-box Attribution via Minimal Interpretable Subset Selection},
  author  = {Chen, Ruoyu and Liang, Siyuan and Li, Jingzhi and Liu, Shiming and Liu, Li and Zhang, Hua and Cao, Xiaochun},
  journal = {arXiv preprint arXiv:2504.00470},
  year    = {2025}
}
```

## License

Released under the [MIT License](LICENSE). Third-party models, datasets, and bundled baselines remain subject to their respective licenses.

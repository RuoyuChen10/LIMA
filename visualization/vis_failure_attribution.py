# -*- coding: utf-8 -*-

"""
Failure attribution visualization for submodular ImageNet results.

This script reads the saved json/npy pairs, skips samples where gt_label and
failure_label are identical, and renders a paper-style panel:
  left top: attribution map
  left bottom: selected searched region
  right: GT/Wrong insertion curves with a vertical marker
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt

matplotlib.use("Agg")


def configure_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Arial", "Liberation Sans", "DejaVu Sans"):
        if font_name in available:
            plt.rc("font", family=font_name)
            return font_name
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize failure attribution insertion curves.")
    parser.add_argument(
        "--explanation-dir",
        type=str,
        default="submodular_results/imagenet-clip-vitl-efficientv2-debug/slico-0.0-0.05-10.0-1.0-pending-samples-8",
        help="Directory containing json/ and npy/ subdirectories.",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Optional single json file to visualize. If omitted, all json files are scanned.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <explanation-dir>/failure_visualization.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="datasets/imagenet/ILSVRC2012_img_val",
        help="Original ImageNet validation image directory. Used only as a fallback.",
    )
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=0.005,
        help="Tolerance for treating nearby GT insertion scores as the peak plateau.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of rendered figures for batch mode.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Saved figure DPI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_to_npy_path(json_path, explanation_dir):
    json_path = Path(json_path)
    explanation_dir = Path(explanation_dir)
    try:
        rel = json_path.relative_to(explanation_dir / "json")
        return explanation_dir / "npy" / rel.with_suffix(".npy")
    except ValueError:
        class_id = json_path.parent.name
        return explanation_dir / "npy" / class_id / json_path.with_suffix(".npy").name


def find_image(json_path, image_root):
    stem = Path(json_path).stem
    for suffix in (".JPEG", ".jpg", ".jpeg", ".png"):
        candidate = Path(image_root) / f"{stem}{suffix}"
        if candidate.exists():
            image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.resize(image, (224, 224))
    return None


def normalize01(array):
    array = np.asarray(array, dtype=np.float32)
    amin = float(np.min(array))
    amax = float(np.max(array))
    if amax <= amin:
        return np.zeros_like(array, dtype=np.float32)
    return (array - amin) / (amax - amin)


def make_attribution_map(image_bgr, masks, gt_scores, wrong_scores):
    margin = np.asarray(gt_scores, dtype=np.float32) - np.asarray(wrong_scores, dtype=np.float32)
    prev_margin = np.concatenate(([0.0], margin[:-1]))
    gain = margin - prev_margin

    heat = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    for mask_img, value in zip(masks[: len(gain)], gain):
        region = mask_img.sum(axis=-1) > 0
        heat[region] = value

    heat = normalize01(heat)
    heat_u8 = np.uint8(255 * heat)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_COOL)
    cam = cv2.addWeighted(image_bgr.astype(np.float32), 0.52, color.astype(np.float32), 0.48, 0.0)
    return np.clip(cam, 0, 255).astype(np.uint8)


def cumulative_insertions(image_bgr, masks):
    insertion = np.zeros_like(image_bgr)
    images = [insertion.copy()]
    ratios = [0.0]

    for mask_img in masks:
        insertion = np.maximum(insertion, mask_img)
        images.append(insertion.copy())
        ratios.append(float((insertion.sum(axis=-1) > 0).sum()) / float(image_bgr.shape[0] * image_bgr.shape[1]))

    return images, np.asarray(ratios, dtype=np.float32)


def score_curves(data):
    gt_scores = list(map(float, data["consistency_score"]))
    wrong_scores = list(map(float, data["failure_score"]))
    n = min(len(gt_scores), len(wrong_scores))
    gt_scores = gt_scores[:n]
    wrong_scores = wrong_scores[:n]

    if "original_gt_score" in data and "original_failure_score" in data:
        gt_scores = [float(data["original_gt_score"])] + gt_scores
        wrong_scores = [float(data["original_failure_score"])] + wrong_scores

    return np.asarray(gt_scores, dtype=np.float32), np.asarray(wrong_scores, dtype=np.float32)


def choose_step(gt_scores, drop_tolerance):
    peak = float(np.max(gt_scores))
    peak_candidates = np.where(gt_scores >= peak - drop_tolerance)[0]
    peak_candidates = peak_candidates[peak_candidates > 0]
    if len(peak_candidates) > 0:
        return int(peak_candidates[-1])
    return int(np.argmax(gt_scores))


def selected_region_panel(image_bgr, insertion_bgr):
    selected = insertion_bgr.sum(axis=-1) > 0
    panel = image_bgr.copy().astype(np.float32)
    panel[~selected] *= 0.28

    if selected.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(selected.astype(np.uint8), kernel, iterations=3).astype(bool)
        edge = dilated & ~selected
        panel[edge] = np.array([30, 30, 235], dtype=np.float32)

    return np.clip(panel, 0, 255).astype(np.uint8)


def style_image_axis(ax, title):
    ax.set_title(title, fontsize=32, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_figure(json_path, npy_path, output_path, image_root, drop_tolerance, dpi):
    data = load_json(json_path)
    if "gt_label" not in data or "failure_label" not in data:
        return "skip_missing_labels"
    if data["gt_label"] == data["failure_label"]:
        return "skip_same_label"
    if "consistency_score" not in data or "failure_score" not in data:
        return "skip_missing_scores"
    if not Path(npy_path).exists():
        return "skip_missing_npy"

    masks = np.load(npy_path)
    masks = masks.astype(np.uint8)
    image_bgr = find_image(json_path, image_root)
    if image_bgr is None:
        image_bgr = np.clip(masks.sum(axis=0), 0, 255).astype(np.uint8)
    else:
        image_bgr = cv2.resize(image_bgr, (masks.shape[2], masks.shape[1]))

    gt_scores, wrong_scores = score_curves(data)
    n_steps = min(len(gt_scores) - 1, len(masks))
    if n_steps <= 0:
        return "skip_empty_scores"

    masks = masks[:n_steps]
    gt_scores = gt_scores[: n_steps + 1]
    wrong_scores = wrong_scores[: n_steps + 1]
    insertions, x = cumulative_insertions(image_bgr, masks)
    x = x[: len(gt_scores)]

    selected_step = choose_step(gt_scores, drop_tolerance)
    selected_step = max(0, min(selected_step, len(insertions) - 1))

    attr_map = make_attribution_map(image_bgr, masks, gt_scores[1:], wrong_scores[1:])
    region_panel = selected_region_panel(image_bgr, insertions[selected_step])

    gt_name = data.get("gt_class_name", str(data["gt_label"]))
    wrong_name = data.get("failure_class_name", str(data["failure_label"]))

    fig = plt.figure(figsize=(14.5, 8.0), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.72],
        height_ratios=[1.0, 1.0],
        wspace=0.32,
        hspace=0.28,
    )

    ax_attr = fig.add_subplot(gs[0, 0])
    ax_region = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[:, 1])

    style_image_axis(ax_attr, "Attribution Map")
    ax_attr.imshow(attr_map[..., ::-1])

    style_image_axis(ax_region, "Searched Region")
    ax_region.imshow(region_panel[..., ::-1])
    ax_region.set_xlabel(
        "Revealed {:.1f}%".format(100.0 * x[selected_step]),
        fontsize=24,
        labelpad=8,
    )

    gt_color = "#1b9e4b"
    wrong_color = "#ef8a8a"
    ax_curve.plot(x, gt_scores, color=gt_color, linewidth=4.0, label=f"GT: {gt_name}")
    ax_curve.plot(x, wrong_scores, color=wrong_color, linewidth=4.0, label=f"Wrong: {wrong_name}")
    ax_curve.fill_between(x, gt_scores, color=gt_color, alpha=0.10)
    ax_curve.fill_between(x, wrong_scores, color=wrong_color, alpha=0.10)
    ax_curve.axvline(x=x[selected_step], color="red", linewidth=3.8)
    ax_curve.scatter([x[selected_step]], [gt_scores[selected_step]], color=gt_color, s=90, zorder=4)
    ax_curve.scatter([x[selected_step]], [wrong_scores[selected_step]], color=wrong_color, s=90, zorder=4)

    ax_curve.set_xlim(0.0, max(1.0, float(x.max())))
    ax_curve.set_ylim(0.0, 1.02)
    ax_curve.set_title("Insertion", fontsize=34, pad=12)
    ax_curve.set_xlabel("Percentage of image revealed", fontsize=32, labelpad=12)
    ax_curve.set_ylabel("Recognition Score", fontsize=32, labelpad=12)
    ax_curve.tick_params(axis="both", which="major", labelsize=27, width=3.2, length=8)
    ax_curve.grid(False)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.spines["bottom"].set_linewidth(3.2)
    ax_curve.spines["left"].set_linewidth(3.2)
    ax_curve.legend(loc="upper right", fontsize=22, frameon=False, handlelength=2.6)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return "rendered"


def iter_json_files(explanation_dir, json_file):
    if json_file is not None:
        yield Path(json_file)
        return

    json_root = Path(explanation_dir) / "json"
    for path in sorted(json_root.glob("*/*.json")):
        yield path


def main(args):
    font_name = configure_font()
    if font_name is not None:
        print(f"Using font: {font_name}")

    explanation_dir = Path(args.explanation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else explanation_dir / "failure_visualization"
    ensure_dir(output_dir)

    counts = {}
    rendered = 0
    for json_path in iter_json_files(explanation_dir, args.json_file):
        npy_path = json_to_npy_path(json_path, explanation_dir)
        out_class_dir = output_dir / json_path.parent.name
        ensure_dir(out_class_dir)
        output_path = out_class_dir / json_path.with_suffix(".png").name
        if output_path.exists() and not args.overwrite:
            status = "skip_exists"
        else:
            status = render_figure(
                json_path=json_path,
                npy_path=npy_path,
                output_path=output_path,
                image_root=args.image_root,
                drop_tolerance=args.drop_tolerance,
                dpi=args.dpi,
            )
        counts[status] = counts.get(status, 0) + 1
        if status == "rendered":
            rendered += 1
            print(f"[rendered] {output_path}")
        if args.max_samples is not None and rendered >= args.max_samples:
            break

    print("Summary:", counts)


if __name__ == "__main__":
    main(parse_args())

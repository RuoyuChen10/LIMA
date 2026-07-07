# -*- coding: utf-8 -*-

"""
Diverging-color failure attribution visualization.

This keeps visualization/vis_failure_attribution.py untouched and writes figures
to <explanation-dir>/failure_visualization_diverging by default.
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

import vis_failure_attribution as base

matplotlib.use("Agg")


def make_failure_colormap():
    return mcolors.LinearSegmentedColormap.from_list(
        "wrong_neutral_gt",
        [
            (0.00, "#ef8a8a"),
            (0.50, "#f7f7f7"),
            (1.00, "#1b9e4b"),
        ],
    )


def make_diverging_attribution_map(image_bgr, masks, selected_step, cmap):
    steps = np.arange(1, len(masks) + 1, dtype=np.float32)
    values = np.zeros_like(steps, dtype=np.float32)
    before = steps <= float(selected_step)
    after = steps > float(selected_step)
    if before.any():
        before_denom = max(float(selected_step - 1), 1.0)
        values[before] = (float(selected_step) - steps[before]) / before_denom
    if after.any():
        after_denom = max(float(len(masks) - selected_step), 1.0)
        values[after] = -(steps[after] - float(selected_step)) / after_denom
    values = np.clip(values, -1.0, 1.0)

    heat = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    for mask_img, value in zip(masks, values):
        region = mask_img.sum(axis=-1) > 0
        heat[region] = value

    color_rgb = np.uint8(255 * cmap((heat + 1.0) / 2.0)[..., :3])
    color_bgr = color_rgb[..., ::-1]
    cam = cv2.addWeighted(image_bgr.astype(np.float32), 0.38, color_bgr.astype(np.float32), 0.62, 0.0)
    return np.clip(cam, 0, 255).astype(np.uint8)


def add_green_region_edge(image_bgr, insertion_bgr):
    selected = insertion_bgr.sum(axis=-1) > 0
    panel = image_bgr.copy()
    if selected.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge = cv2.dilate(selected.astype(np.uint8), kernel, iterations=2).astype(bool) & ~selected
        panel[edge] = np.array([45, 170, 35], dtype=np.uint8)
    return panel


def render_figure(json_path, npy_path, output_path, image_root, drop_tolerance, dpi):
    data = base.load_json(json_path)
    if "gt_label" not in data or "failure_label" not in data:
        return "skip_missing_labels"
    if data["gt_label"] == data["failure_label"]:
        return "skip_same_label"
    if "consistency_score" not in data or "failure_score" not in data:
        return "skip_missing_scores"
    if not Path(npy_path).exists():
        return "skip_missing_npy"

    masks = np.load(npy_path).astype(np.uint8)
    image_bgr = base.find_image(json_path, image_root)
    if image_bgr is None:
        image_bgr = np.clip(masks.sum(axis=0), 0, 255).astype(np.uint8)
    else:
        image_bgr = cv2.resize(image_bgr, (masks.shape[2], masks.shape[1]))

    gt_scores, wrong_scores = base.score_curves(data)
    n_steps = min(len(gt_scores) - 1, len(masks))
    if n_steps <= 0:
        return "skip_empty_scores"

    masks = masks[:n_steps]
    gt_scores = gt_scores[: n_steps + 1]
    wrong_scores = wrong_scores[: n_steps + 1]
    insertions, x = base.cumulative_insertions(image_bgr, masks)
    x = x[: len(gt_scores)]

    selected_step = base.choose_step(gt_scores, drop_tolerance)
    selected_step = max(0, min(selected_step, len(insertions) - 1))

    cmap = make_failure_colormap()
    attr_map = make_diverging_attribution_map(image_bgr, masks, selected_step, cmap)
    attr_map = add_green_region_edge(attr_map, insertions[selected_step])
    region_panel = base.selected_region_panel(image_bgr, insertions[selected_step])

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

    base.style_image_axis(ax_attr, "Attribution Map")
    ax_attr.imshow(attr_map[..., ::-1])
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = ax_attr.inset_axes([-0.09, 0.0, 0.045, 1.0])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([-1.0, 1.0])
    cbar.set_ticklabels(["W", "GT"])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.tick_params(labelsize=19, width=2.0, length=5)
    cbar.outline.set_linewidth(1.8)

    base.style_image_axis(ax_region, "Searched Region")
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
    ax_curve.axvline(x=x[selected_step], color=gt_color, linewidth=3.4, linestyle="--")
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


def main(args):
    font_name = base.configure_font()
    if font_name is not None:
        print(f"Using font: {font_name}")

    explanation_dir = Path(args.explanation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else explanation_dir / "failure_visualization_diverging"
    base.ensure_dir(output_dir)

    counts = {}
    rendered = 0
    for json_path in base.iter_json_files(explanation_dir, args.json_file):
        npy_path = base.json_to_npy_path(json_path, explanation_dir)
        out_class_dir = output_dir / json_path.parent.name
        base.ensure_dir(out_class_dir)
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
    main(base.parse_args())

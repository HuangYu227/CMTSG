from __future__ import annotations

from pathlib import Path

import numpy as np

from cmtsg.utils import resolve_path


def _render_with_pillow(values: np.ndarray, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1120, 520
    margin = 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    draw.rectangle([margin, margin, width - margin, height - margin], outline=(220, 220, 220))

    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    span = max(v_max - v_min, 1e-6)
    x_scale = plot_w / max(1, values.shape[0] - 1)
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    for var_idx in range(values.shape[1]):
        points = []
        for t_idx in range(values.shape[0]):
            x = margin + t_idx * x_scale
            y = height - margin - ((float(values[t_idx, var_idx]) - v_min) / span) * plot_h
            points.append((x, y))
        color = palette[var_idx % len(palette)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2 if values.shape[1] <= 8 else 1)
    image.save(output_path)


def render_line_chart(sample: np.ndarray, output_path: str | Path, title: str | None = None) -> Path:
    values = np.asarray(sample, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected sample shape [L,K], got {values.shape}")

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _render_with_pillow(values, output_path)
        return output_path

    fig_width = 8.0
    fig_height = 3.2 if values.shape[1] <= 4 else 4.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=140)
    x = np.arange(values.shape[0])
    for var_idx in range(values.shape[1]):
        alpha = 0.9 if values.shape[1] <= 8 else 0.55
        linewidth = 1.5 if values.shape[1] <= 8 else 0.8
        ax.plot(x, values[:, var_idx], linewidth=linewidth, alpha=alpha)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

from __future__ import annotations

from typing import Sequence
import warnings
import numpy as np

import open3d.visualization.gui as gui


def suppress_requests_dependency_warning() -> None:
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except Exception:
        warnings.filterwarnings(
            "ignore",
            message=r".*doesn't match a supported version!",
        )


def normalize_scalar(values: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> tuple[np.ndarray, float, float]:
    lo = float(np.percentile(values, low_q))
    hi = float(np.percentile(values, high_q))
    if hi <= lo:
        hi = lo + 1e-12
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return norm.astype(np.float32), lo, hi


def scalar_to_colors(norm: np.ndarray) -> np.ndarray:
    return np.stack([norm, 0.2 * np.ones_like(norm), 1.0 - norm], axis=1).astype(np.float32)


class LegendPanel:
    def __init__(self, legend_entries: Sequence[tuple[str, Sequence[float]]], max_height: int = 400) -> None:
        self._legend_entries = list(legend_entries)
        self._max_height = max_height

        self.widget = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        self.widget.background_color = gui.Color(0.05, 0.05, 0.05, 0.85)

        title = gui.Label("   Legend")
        title.text_color = gui.Color(0.9, 0.9, 0.9, 1.0)
        self.widget.add_child(title)

        self._scroll = gui.ScrollableVert(2, gui.Margins(6, 4, 6, 6))
        for name, color in self._legend_entries:
            row = gui.Horiz(6)

            swatch = gui.Button("  ")
            swatch.background_color = gui.Color(float(color[0]), float(color[1]), float(color[2]), 1.0)
            swatch.toggleable = False

            label = gui.Label(name)
            label.text_color = gui.Color(0.95, 0.95, 0.95, 1.0)

            row.add_child(swatch)
            row.add_child(label)
            self._scroll.add_child(row)

        self.widget.add_child(self._scroll)

    def layout(self, content_rect: gui.Rect, layout_context: gui.LayoutContext) -> None:
        em = layout_context.theme.font_size
        longest = max((len(name) for name, _ in self._legend_entries), default=0)
        panel_width = max(220, int(longest * em * 0.65 + 60))
        panel_height = min(self._max_height, content_rect.height - 20)

        self.widget.frame = gui.Rect(
            content_rect.get_right() - panel_width - 10,
            content_rect.y + 10,
            panel_width,
            panel_height,
        )


class CurvatureGradientLegendPanel:
    def __init__(self, vmin: float, vmax: float, steps: int = 24, title: str = "Curvature") -> None:
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._steps = max(8, int(steps))

        self.widget = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        self.widget.background_color = gui.Color(0.05, 0.05, 0.05, 0.85)

        self._title_label = gui.Label(f"   {title}")
        self._title_label.text_color = gui.Color(0.9, 0.9, 0.9, 1.0)
        self.widget.add_child(self._title_label)

        self._max_label = gui.Label(f"max: {self._vmax:.6f}")
        self._max_label.text_color = gui.Color(0.95, 0.95, 0.95, 1.0)
        self.widget.add_child(self._max_label)

        bar = gui.Vert(0, gui.Margins(8, 4, 8, 4))
        for t in np.linspace(1.0, 0.0, self._steps):
            swatch = gui.Button("   ")
            swatch.background_color = gui.Color(float(t), 0.2, float(1.0 - t), 1.0)
            swatch.toggleable = False
            bar.add_child(swatch)
        self.widget.add_child(bar)

        self._min_label = gui.Label(f"min: {self._vmin:.6f}")
        self._min_label.text_color = gui.Color(0.95, 0.95, 0.95, 1.0)
        self.widget.add_child(self._min_label)

    def set_title(self, title: str) -> None:
        self._title_label.text = f"   {title}"

    def set_range(self, vmin: float, vmax: float) -> None:
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._max_label.text = f"max: {self._vmax:.6f}"
        self._min_label.text = f"min: {self._vmin:.6f}"

    def layout(self, content_rect: gui.Rect, _layout_context: gui.LayoutContext) -> None:
        panel_width = 220
        panel_height = min(520, content_rect.height - 20)
        self.widget.frame = gui.Rect(
            content_rect.get_right() - panel_width - 10,
            content_rect.y + 10,
            panel_width,
            panel_height,
        )

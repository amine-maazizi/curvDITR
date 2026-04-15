from __future__ import annotations

from typing import Sequence
import warnings

import open3d.visualization.gui as gui


def suppress_requests_dependency_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 .*or chardet .*doesn't match a supported version!",
    )


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

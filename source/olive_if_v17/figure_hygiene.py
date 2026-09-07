"""Figure hygiene for every matplotlib figure the paper ships
(checklist items: no in-figure titles; panel letters (a), (b), … *below*
each panel; legend never overlaps data; Type-42 fonts).

Use as a context manager around an existing figure script, or call
``finalize(fig)`` before ``savefig``::

    from olive_if_v17.figure_hygiene import finalize, install_savefig_hook
    install_savefig_hook(report_dir="artifacts_v17")   # every plt.savefig is cleaned
    ...
    plt.savefig("fig_v3_1_capacity.pdf")               # titles gone, (a)(b) added, legend moved

Legend placement is decided by measuring the intersection of the legend's
bounding box with every Line2D / Patch / PathCollection / Text in the axes
after a draw; candidate locations are tried in order and, when every inside
location overlaps, the legend is moved outside the axes (right or below).
A per-legend record (overlap before/after, final location) is appended to
``legend_overlap_report_v17.csv`` — that report is what the checklist audit
reads for item C23.
"""
from __future__ import annotations

import csv
import pathlib
import string

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

LOCS = ["upper right", "upper left", "lower right", "lower left", "center right",
        "center left", "upper center", "lower center"]
_REPORT: pathlib.Path | None = None
_ORIG_SAVEFIG = None


# ------------------------------------------------------------- titles
def strip_titles(fig) -> int:
    n = 0
    if fig._suptitle is not None and fig._suptitle.get_text():
        fig._suptitle.set_text(""); n += 1
    for ax in fig.axes:
        if ax.get_title():
            ax.set_title(""); n += 1
        for pos in ("left", "right"):
            if ax.get_title(loc=pos):
                ax.set_title("", loc=pos); n += 1
    return n


# ------------------------------------------------------------- panel letters
def label_panels(fig, offset: float = -0.22, fontsize: int = 10, only_if_multi: bool = True) -> int:
    """Put '(a)', '(b)', … centred *below* each panel (never inside the axes)."""
    axes = [ax for ax in fig.axes if ax.get_visible() and not getattr(ax, "_colorbar", False)
            and ax.get_label() != "<colorbar>"]
    seen, uniq = set(), []
    for ax in axes:                                  # twinx/twiny share a position: one letter
        key = tuple(round(v, 3) for v in ax.get_position().bounds)
        if key not in seen:
            seen.add(key); uniq.append(ax)
    axes = uniq
    if only_if_multi and len(axes) < 2:
        return 0
    axes = sorted(axes, key=lambda a: (-round(a.get_position().y0, 2), a.get_position().x0))
    for ax, letter in zip(axes, string.ascii_lowercase):
        # remove a letter placed inside the axes by an older script
        for t in list(ax.texts):
            if t.get_text().strip("() ") in string.ascii_lowercase and len(t.get_text()) <= 3:
                t.remove()
        ax.text(0.5, offset, f"({letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=fontsize, clip_on=False)
    return len(axes)


# ------------------------------------------------------------- legend overlap
def _artist_points(ax, renderer):
    """Display-space vertices of every data artist (lines, patches, collections,
    texts).  A vertex test is used instead of bounding-box intersection because
    an axvline/axhline or a long curve has a bbox that covers the whole axes and
    would make every legend location look occupied."""
    import numpy as np
    pts = []
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) > 1:  # densify so a two-point axhline is still detected along its length
                t = np.linspace(0, 1, 20)[:, None]
                xy = np.concatenate([xy[:-1] + (xy[1:] - xy[:-1]) * ti for ti in t])
            pts.append(ln.get_transform().transform(xy))
    for pa in ax.patches:
        a = pa.get_alpha()
        if a is not None and a < 0.35:
            continue
        try:
            pts.append(pa.get_transform().transform(pa.get_path().vertices))
        except Exception:
            pass
    for co in ax.collections:
        a = co.get_alpha()
        if a is not None and a < 0.35:           # faint shading (axvspan/fill_between) may sit under a legend
            continue
        try:
            off = co.get_offsets()
            if len(off):
                pts.append(co.get_offset_transform().transform(off))
            else:
                for path in co.get_paths()[:200]:
                    pts.append(co.get_transform().transform(path.vertices))
        except Exception:
            pass
    for tx in ax.texts:
        try:
            bb = tx.get_window_extent(renderer)
            pts.append(np.array([[bb.x0, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1], [bb.x1, bb.y0]]))
        except Exception:
            pass
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def _overlap_count(lb: Bbox, pts) -> int:
    if len(pts) == 0:
        return 0
    inside = (pts[:, 0] >= lb.x0) & (pts[:, 0] <= lb.x1) & (pts[:, 1] >= lb.y0) & (pts[:, 1] <= lb.y1)
    return int(inside.sum())


def fix_legend(ax, fig, multi_panel: bool = False) -> dict:
    leg = ax.get_legend()
    if leg is None:
        return {}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = _artist_points(ax, renderer)
    before = _overlap_count(leg.get_window_extent(renderer), pts)
    handles, labels = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles, \
        [t.get_text() for t in leg.get_texts()]
    if not handles:
        return {"overlap_before": before, "overlap_after": before, "loc": "n/a"}
    kw = {"frameon": leg.get_frame_on(), "fontsize": leg._fontsize,
          "ncol": getattr(leg, "_ncols", 1)}
    cur = leg._loc if isinstance(leg._loc, str) else "best"
    best = (before, cur)
    if before > 0:
        for loc in LOCS:
            ax.legend(handles, labels, loc=loc, **kw)
            fig.canvas.draw()
            n = _overlap_count(ax.get_legend().get_window_extent(renderer), pts)
            if n < best[0]:
                best = (n, loc)
            if n == 0:
                break
        if best[0] > 0 and not multi_panel:      # single panel: move it out of the axes entirely
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                      borderaxespad=0, frameon=kw["frameon"], fontsize=kw["fontsize"])
            best = (0, "outside-below")
        elif best[0] > 0:                          # multi panel: smallest font that clears the data
            for fs in (kw["fontsize"] - 1, kw["fontsize"] - 2):
                for loc in LOCS:
                    ax.legend(handles, labels, loc=loc, frameon=kw["frameon"], fontsize=fs, ncol=kw["ncol"])
                    fig.canvas.draw()
                    n = _overlap_count(ax.get_legend().get_window_extent(renderer), pts)
                    if n < best[0]:
                        best = (n, f"{loc}@{fs}pt")
                    if n == 0:
                        break
                if best[0] == 0:
                    break
            if best[0] > 0:
                loc = best[1].split("@")[0]
                ax.legend(handles, labels, loc=loc, **kw)
        else:
            ax.legend(handles, labels, loc=best[1], **kw)
    return {"overlap_before": before, "overlap_after": best[0], "loc": best[1]}


# ------------------------------------------------------------- driver
def finalize(fig, name: str = "", report_dir: str | pathlib.Path | None = None) -> list[dict]:
    """Apply all hygiene rules to ``fig`` and append a report row per legend."""
    strip_titles(fig)
    label_panels(fig)
    rows = []
    multi = len({tuple(round(v, 3) for v in a.get_position().bounds) for a in fig.axes}) > 1
    for i, ax in enumerate(fig.axes):
        r = fix_legend(ax, fig, multi_panel=multi)
        if r:
            r.update({"figure": name, "axes": i})
            rows.append(r)
    rep = pathlib.Path(report_dir) if report_dir else _REPORT
    if rep is not None:
        rep.mkdir(parents=True, exist_ok=True)
        f = rep / "legend_overlap_report_v17.csv"
        new = not f.exists()
        with f.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["figure", "axes", "overlap_before", "overlap_after", "loc"])
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)
            if not rows:  # a figure without legends still counts as inspected
                w.writerow({"figure": name, "axes": -1, "overlap_before": 0, "overlap_after": 0, "loc": "no-legend"})
    return rows


def install_savefig_hook(report_dir: str | pathlib.Path = "artifacts_v17") -> None:
    """Make every ``plt.savefig`` / ``Figure.savefig`` pass through ``finalize``."""
    global _REPORT, _ORIG_SAVEFIG
    _REPORT = pathlib.Path(report_dir)
    if _ORIG_SAVEFIG is not None:
        return
    _ORIG_SAVEFIG = matplotlib.figure.Figure.savefig

    def _hooked(self, fname, *a, **k):
        finalize(self, name=str(fname), report_dir=_REPORT)
        k.setdefault("bbox_inches", "tight")
        return _ORIG_SAVEFIG(self, fname, *a, **k)

    matplotlib.figure.Figure.savefig = _hooked


def run_scripts(scripts: list[pathlib.Path], report_dir: pathlib.Path) -> None:
    """Re-execute existing figure scripts with the hook installed."""
    import runpy
    install_savefig_hook(report_dir)
    for s in scripts:
        print("regenerating through hygiene hook:", s)
        runpy.run_path(str(s), run_name="__main__")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--out", default="artifacts_v17")
    a = ap.parse_args()
    run_scripts([pathlib.Path(s) for s in a.scripts], pathlib.Path(a.out))

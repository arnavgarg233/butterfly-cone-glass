"""Shared matplotlib style for the ButterflyCone "butterfly cone" paper (Nature Physics target).

Every data figure (main Figs 2-5 + Extended Data / SI) imports this module so the
whole display set reads as one visual system that matches the author's own
Nature-style figures in ~/Downloads/QuantDynam_figures/.

Usage
-----
    import figstyle as fs
    fs.use()                                   # set rcParams (also runs on import)

    fig, axes = plt.subplots(1, 3, figsize=fs.figsize(fs.WIDTH_FULL, 0.36))
    ax = axes[0]
    ax.plot(t, D, color=fs.MEASURED)           # semantic colour
    ax.plot(t, fit, color=fs.THEORY, ls="--")
    fs.panel_label(ax, "a")                     # bold lowercase, upper-left

    # ordered/ladder series (T-ladder, kick amplitude, N ...):
    for c, y in zip(fs.sequential(len(series)), series):
        ax.plot(x, y, color=c)

    fs.finalize(fig)
    fs.save(fig, "results/figures/fig2_butterfly_cone")   # writes .pdf AND .png

Palette
-------
Categorical  = Okabe-Ito (the author's figures are built on it: #0072B2, #D55E00,
#009E73 recur in every panel). Colour-blind safe for deuteranopia/protanopia/
tritanopia.
Sequential   = cividis (the navy->khaki colourbar in the author's Fig 4a); perceptually
uniform and colour-blind safe. Use for anything with a natural order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Canvas geometry  (Nature Physics: single column 88 mm, full width 180 mm max)
# --------------------------------------------------------------------------- #
MM: float = 1.0 / 25.4               # millimetres -> inches
WIDTH_FULL: float = 180.0 * MM       # 7.09 in  -- double-column / full-page max
WIDTH_THREEQ: float = 136.0 * MM     # 5.35 in  -- ~3/4 width
WIDTH_SINGLE: float = 88.0 * MM      # 3.46 in  -- single column
GOLDEN: float = 0.618                # a pleasant default single-row aspect (h/w)


def figsize(width: float = WIDTH_FULL, aspect: float = GOLDEN) -> tuple[float, float]:
    """(width_in_inches, width*aspect). Pass a WIDTH_* constant for `width`."""
    return (width, width * aspect)


# --------------------------------------------------------------------------- #
# Palette  --  Okabe-Ito categorical (named list) + semantic aliases
# --------------------------------------------------------------------------- #
# Colour-blind-safe categorical cycle, ordered to match the author's usage
# (blue = measured data, vermillion = theory/highlight, green = secondary/pass).
PALETTE: list[str] = [
    "#0072B2",  # 0 blue          -- measured / primary data
    "#D55E00",  # 1 vermillion    -- theory line, deep anchor, highlight
    "#009E73",  # 2 bluish green  -- secondary axis, pass/gate, special markers
    "#E69F00",  # 3 orange/gold   -- third series
    "#56B4E9",  # 4 sky blue      -- fourth series / light accent
    "#CC79A7",  # 5 reddish purple-- fifth series
    "#0A2A4A",  # 6 deep navy     -- darkest series / emphasis
    "#7F7F7F",  # 7 neutral gray  -- controls / reference / de-emphasis
]

# --- semantic aliases (prefer these over PALETTE[i] in figure code) --------- #
BLUE = MEASURED = PALETTE[0]         # measured observable
VERMILLION = THEORY = ACCENT = PALETTE[1]   # prediction / theory / the headline point
GREEN = HIGHLIGHT = PALETTE[2]       # secondary y-axis, certification/pass, key marker
GOLD = PALETTE[3]
SKY = PALETTE[4]
PURPLE = PALETTE[5]
NAVY = PALETTE[6]
GRAY = CONTROL = PALETTE[7]

INK = "#1A1A1A"        # axes, ticks, primary text (near-black, not pure black)
SUBTLE = "#6E6E6E"     # secondary annotation text, guide lines
GUIDE = "#B3B3B3"      # 1:1 lines, reference gridlines
PANEL_BG = "#FFFFFF"


def _mute(hex_color: str, frac: float = 0.20) -> str:
    """Return `hex_color` desaturated by `frac` toward a luminance-matched gray.

    Drops chroma while holding perceived lightness ~fixed, so a colour keeps its
    hue (colour-blind-safe separations preserved) but reads as the author's more
    muted Okabe-Ito variants rather than the fully-saturated primaries.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mix = lambda c: int(round(c * (1.0 - frac) + lum * frac))  # noqa: E731
    return "#%02X%02X%02X" % (mix(r), mix(g), mix(b))


# Muted variants for the most-saturated reference/secondary marks, nudged ~20%
# toward gray to sit in the author's softer palette (sampled orange ~#C04800,
# green ~#009060).  Use for de-emphasised reference lines / secondary observables.
VERMILLION_MUTED = _mute(VERMILLION, 0.20)   # ~#C16217  muted ceiling / reference line
GREEN_MUTED = _mute(GREEN, 0.20)             # ~#189774  muted secondary (D_sat/N) marks

SEQUENTIAL: str = "cividis"          # named cmap for ordered/ladder series
SEQUENTIAL_R: str = "cividis_r"


def sequential(n: int, lo: float = 0.08, hi: float = 0.92,
               cmap: str = SEQUENTIAL) -> list[tuple]:
    """`n` colours sampled along `cmap`, trimmed off the too-dark/too-light ends.

    Order convention for the paper: index 0 = coldest/deepest state (navy),
    index n-1 = warmest/shallowest (khaki). Reverse the input if you need the
    opposite mapping.
    """
    if n <= 0:
        return []
    if n == 1:
        return [mpl.colormaps[cmap](0.5)]
    xs = np.linspace(lo, hi, n)
    return [mpl.colormaps[cmap](x) for x in xs]


# --------------------------------------------------------------------------- #
# Fonts  --  clean sans-serif body; STIX-sans mathtext with italic variables but
# UPRIGHT roman digits/operators (matches the author: italicise only variables)
# --------------------------------------------------------------------------- #
def _first_available(prefs: Sequence[str], default: str) -> str:
    have = {f.name for f in _fm.fontManager.ttflist}
    for p in prefs:
        if p in have:
            return p
    return default


# Arial first for reliable bold-weight embedding; Helvetica/DejaVu as fallbacks.
_SANS = _first_available(
    ["Arial", "Helvetica", "Helvetica Neue", "TeX Gyre Heros", "DejaVu Sans"],
    "DejaVu Sans",
)

# --------------------------------------------------------------------------- #
# Type scale (points).  Base 8 pt is the Nature-figure sweet spot at 180 mm.
# --------------------------------------------------------------------------- #
FS_BASE = 8.0
FS_LABEL = 9.0        # axis labels
FS_TICK = 8.0         # tick labels
FS_LEGEND = 7.5
FS_ANNOT = 7.5        # in-panel annotation / stat blocks
FS_PANEL = 11.0       # bold panel letter (a, b, c ...)


def use() -> None:
    """Install the shared rcParams. Called automatically on import; call again
    to undo any local `plt.rc` overrides made mid-session."""
    mpl.rcParams.update({
        # ---- fonts ----
        "font.family": "sans-serif",
        "font.sans-serif": [_SANS, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS_BASE,
        # STIX sans math follows the TeX/author convention: variables (letters)
        # italic, but NUMBERS and operators upright roman -- unlike dejavusans,
        # which italicises digits too.  Keeps prose upright, math variables italic.
        "mathtext.fontset": "stixsans",     # sans math, roman digits (author's look)
        "mathtext.default": "it",           # letters italic; digits stay roman
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,

        # ---- figure / export ----
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": PANEL_BG,
        "savefig.facecolor": PANEL_BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
        "pdf.fonttype": 42,                 # embed editable TrueType (Nature req.)
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # ---- axes ----
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.labelpad": 3.0,
        "axes.spines.top": False,           # clean: no top / right spine
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.autolimit_mode": "data",

        # ---- ticks (outward, left+bottom only) ----
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.top": False,
        "ytick.right": False,

        # ---- lines / markers ----
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.5,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "errorbar.capsize": 2.0,

        # ---- legend (frameless) ----
        "legend.frameon": False,
        "legend.fontsize": FS_LEGEND,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.4,

        # ---- grid (off by default; enable per-axes if needed) ----
        "grid.color": GUIDE,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "axes.grid": False,
    })


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04,
                weight: str = "bold", size: float = FS_PANEL, **kw):
    """Bold lowercase panel tag (a, b, c ...) at the axes' upper-left corner.

    Positioned in axes-fraction coordinates just outside the plot box, matching
    the author's convention. Nudge `x` (more negative when the y-axis carries a
    wide tick label / long axis title) and `y` as needed per panel.
    """
    return ax.text(x, y, letter, transform=ax.transAxes,
                   fontsize=size, fontweight=weight, va="bottom", ha="left",
                   color=INK, clip_on=False, **kw)


def annotate_stats(ax, text: str, x: float = 0.04, y: float = 0.96,
                   color: str = INK, size: float = FS_ANNOT, **kw):
    """In-panel stat block (slope / R^2 / n ...) top-left, as the author does.
    Use `color=fs.SUBTLE` for secondary notes."""
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=size,
                   color=color, va="top", ha="left", linespacing=1.35, **kw)


def scorecard(ax, rows, x: float = 0.55, y: float = 0.80,
              title: str = "scorecard", *, box_w: float = 0.45,
              bar_w: float = 0.17, row_dy: float = 0.076, bar_h: float = 0.05,
              title_size: float = FS_ANNOT, label_size: float = FS_ANNOT - 2.0,
              count_size: float = FS_ANNOT - 1.7, box: bool = True, zorder: int = 12):
    """Draw the author's signature "forward scorecard" motif in empty panel space.

    A small titled block with a stack of short labelled colour chips (solid bars),
    each carrying a white in-bar category label with its printed count to the
    right -- exactly the author's ``forward scorecard`` device.  Prose is upright
    roman; anything wrapped in ``$...$`` (e.g. ``$R^2$``) stays italic math,
    matching the house style.

    Parameters
    ----------
    rows : list of ``(label, count, color)``
        ``label`` prints inside the solid chip (white, kept short so it fits);
        ``count`` prints to its right (ink); ``color`` is the chip colour.
    x, y : float
        Axes-fraction coordinates of the block's top-left (the title anchor).

    All coordinates are axes fractions, so place the block in a data-free corner.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    n = len(rows)
    top = y + 0.052
    bottom = y - 0.014 - n * row_dy
    if box:
        ax.add_patch(FancyBboxPatch(
            (x - 0.02, bottom), box_w, top - bottom,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            transform=ax.transAxes, facecolor="#F6F6F6", edgecolor="#D9D9D9",
            linewidth=0.6, zorder=zorder, clip_on=False))
    ax.text(x, y + 0.006, title, transform=ax.transAxes, fontsize=title_size,
            fontweight="bold", color=INK, va="bottom", ha="left",
            zorder=zorder + 2, clip_on=False)

    count_x = x + bar_w + 0.014
    for i, (label, count, color) in enumerate(rows):
        yc = y - 0.02 - (i + 0.5) * row_dy
        # solid colour chip
        ax.add_patch(Rectangle((x, yc - bar_h / 2), bar_w, bar_h,
                               transform=ax.transAxes, facecolor=color,
                               edgecolor="none", zorder=zorder + 1, clip_on=False))
        # in-bar label (white, upright prose; math stays italic)
        ax.text(x + 0.010, yc, label, transform=ax.transAxes,
                fontsize=label_size, color="white", va="center", ha="left",
                zorder=zorder + 3, clip_on=False)
        # printed count to the right
        ax.text(count_x, yc, count, transform=ax.transAxes, fontsize=count_size,
                color=INK, va="center", ha="left", zorder=zorder + 3, clip_on=False)


def hide_spines(ax, which: Sequence[str] = ("top", "right")) -> None:
    """Belt-and-braces spine removal (rcParams already drops top/right)."""
    for s in which:
        ax.spines[s].set_visible(False)


def unity_line(ax, color: str = GUIDE, lw: float = 0.8, ls: str = "--", **kw):
    """Draw a y=x reference across the current axes limits (for 1:1 scatters)."""
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    return ax.plot([lo, hi], [lo, hi], color=color, lw=lw, ls=ls, zorder=0, **kw)


def finalize(fig) -> None:
    """Final touch-ups before saving. constrained_layout does the spacing, so
    this is mostly a hook / no-op kept for call-site symmetry."""
    # If a figure opted out of constrained_layout, fall back to tight_layout.
    try:
        if not fig.get_constrained_layout():
            fig.tight_layout()
    except Exception:
        pass


def save(fig, stem: str, formats: Sequence[str] = ("pdf", "png"),
         dpi: int = 300) -> list[Path]:
    """Save `fig` to `stem.<ext>` for each format (vector PDF for the journal,
    PNG for quick viewing). Returns the written paths."""
    out: list[Path] = []
    p = Path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        dest = p.with_suffix(f".{ext}")
        fig.savefig(dest, dpi=dpi)
        out.append(dest)
    return out


# Install on import so `import figstyle` is enough.
use()

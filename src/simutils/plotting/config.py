import matplotlib as mpl
from cycler import cycler


COL_WIDTHS = {
    "single":3.25,
    "double":7.0,
}

COLOR_CYCLES = {
    "default":{"blue":"#1F77B4", "orange":"#FF7F0E", "green":"#2CA02C", "red":"#D62728", "purple":"#9467BD", "brown":"#8C564B", "pink":"#E377C2", "grey":"#7F7F7F", "olive":"#BCBD22", "cyan":"#17BECF"},
    "tol_bright":{"blue":"#4477AA", "yellow":"#CCBB44", "green":"#228833", "red":"#EE6677", "purple":"#AA3377", "cyan":"#66CCEE", "grey":"#BBBBBB"},
    "tol_vibrant":{"blue":"#0077BB", "orange":"#EE7733", "teal":"#009988", "cyan":"#33BBEE", "red":"#CC3311", "magenta":"#EE3377", "grey":"#BBBBBB"},
    "tol_light":{"light blue":"#77AADD", "orange":"#EE8866", "mint":"#44BB99", "light yellow":"#EEDD88", "pink":"#FFAABB", "light cyan":"#99DDFF", "pear":"#BBCC33", "olive":"#AAAA00", "pale grey":"#DDDDDD"},
    "stf":{"red":"#C31014", "green":"#74AA89", "blue":"#4464AD", "yellow":"#FEC12C"},
}

def set_plot_style(layout=None, aspect_ratio=4/3, figsize=None, color_cycle="tol_vibrant", base_font=12, scale=False):
    """
    Set a standard matplotlib plot style.

    Parameters
    ----------
    layout : str, optional
        Plot layout "single" or "double. Defaults to matplotlib default.
    aspect_ratio : float, optional
        Plot aspect ratio (default = 4/3).
    figsize : tuple, optional
        Figure size (width, height).
    color_cycle : str, optional
        "default", "tol_bright", "tol_vibrant", "tol_light", "stf". Defaults to "tol_vibrant".
    base_font : float, optional
        Vase font size (default = 12).
    scale : bool, optional
        Scale fonts and linewidth with figsize (default = False).
    """
    if figsize is not None:
        width, height = figsize
    elif layout is not None:
        if layout not in COL_WIDTHS:
            raise ValueError(f"Invalid layout: '{layout}'. Choose from {list(COL_WIDTHS.keys())}")
        width = COL_WIDTHS[layout]
        height = width / aspect_ratio
    else:
        width, height = mpl.rcParams["figure.figsize"]

    color_cycle_dict = COLOR_CYCLES.get(color_cycle)
    if color_cycle_dict is None:
        raise ValueError(f"Invalid color_cycle name: '{color_cycle}'. Choose from {list(COLOR_CYCLES.keys())}")
    colors = list(color_cycle_dict.values())

    if scale:
        scale = width / COL_WIDTHS["double"]
    else:
        scale = 1.0

    # lines
    mpl.rcParams["lines.linewidth"] = 1.5 * scale
    mpl.rcParams["lines.markersize"] = 6 * scale

    # font
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = base_font * scale

    # axes
    mpl.rcParams["axes.titlesize"] = base_font * scale
    mpl.rcParams["axes.labelsize"] = base_font * scale
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)

    # ticks
    mpl.rcParams["xtick.labelsize"] = base_font * scale
    mpl.rcParams["ytick.labelsize"] = base_font * scale

    # legend
    mpl.rcParams["legend.frameon"] = False
    #mpl.rcParams["legend.framealpha"] = 0.8
    mpl.rcParams["legend.fontsize"] = base_font * scale

    # figure
    mpl.rcParams["figure.figsize"] = (width, height)
    mpl.rcParams["figure.dpi"] = 600
    mpl.rcParams["figure.max_open_warning"] = 20

    # saving figures
    mpl.rcParams["savefig.dpi"] = 600
    mpl.rcParams["savefig.format"] = "png"
    mpl.rcParams["savefig.bbox"] = "tight"

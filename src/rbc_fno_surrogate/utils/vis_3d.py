import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Literal
import os
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)


def set_size(width_pt=455, fraction=1, aspect=0.62):
    inches_per_pt = 1 / 72.27  # Convert pt to inch
    fig_width_in = width_pt * inches_per_pt * fraction  # width in inches
    fig_height_in = fig_width_in * aspect  # height in inches
    return (fig_width_in, fig_height_in)


def plot_paper(gt, pred, anim_dir: str, index: str):
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "DejaVu Serif",
                "CMU Serif",
                "Times New Roman",
            ],
            "axes.labelsize": 9,
            "font.size": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    C, T, H, W, D = gt.shape

    dims = (2 * np.pi, 2 * np.pi, 2)

    # choose time indices (kept explicit as per your edit)
    ncols = 4
    times_idx = [0, 9, 49, 99]
    times_lbl = [1, 10, 50, 100]

    # channel-dependent color limits for the field
    vmin, vmax = 0.0, 1.0
    cmap = "rainbow"

    # compute diff color limits symmetrically over the selected frames
    diff = pred - gt
    # stack the selected frames for robust range computation
    diff_stack = np.stack([diff[0, ti, :, :, :] for ti in times_idx], axis=0)
    diff_abs_max = float(np.nanmax(np.abs(diff_stack)))
    diff_abs_max = max(diff_abs_max, 1e-8)  # avoid zero range
    dvmin, dvmax = -diff_abs_max, diff_abs_max
    cmap_diff = "RdBu_r"

    # figure layout: 3 rows of data (GT, Pred, Diff) × ncols, plus 1 column for colorbars
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=set_size(455, fraction=1, aspect=0.62), dpi=300)
    gs = GridSpec(
        3,
        ncols + 1,
        width_ratios=[1] * ncols + [0.1],
        wspace=0.05,
        hspace=-0.25,
        figure=fig,
    )

    axes_gt, axes_pred, axes_diff = [], [], []
    last_faces_field = None
    last_faces_diff = None

    elev = 15
    azim = -45

    # ---- Row 1: Ground truth ----
    for ci, t_idx in enumerate(times_idx):
        ax = fig.add_subplot(gs[0, ci], projection="3d")
        ax.set_xlim(0, dims[0])
        ax.set_ylim(0, dims[1])
        ax.set_zlim(0, dims[2])
        try:
            ax.set_box_aspect(dims)
        except Exception:
            pass
        ax.set_axis_off()
        faces = plot_cube_faces(
            gt[0, t_idx, :, :, :],
            ax,
            dims=dims,
            contour_levels=50,
            show_back_faces=False,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(rf"$t={times_lbl[ci]}$", pad=4)
        axes_gt.append(ax)
        last_faces_field = faces  # for field colorbar

    # ---- Row 2: Prediction ----
    for ci, t_idx in enumerate(times_idx):
        ax = fig.add_subplot(gs[1, ci], projection="3d")
        ax.set_xlim(0, dims[0])
        ax.set_ylim(0, dims[1])
        ax.set_zlim(0, dims[2])
        try:
            ax.set_box_aspect(dims)
        except Exception:
            pass
        ax.set_axis_off()
        faces = plot_cube_faces(
            pred[0, t_idx, :, :, :],
            ax,
            dims=dims,
            contour_levels=50,
            show_back_faces=False,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.view_init(elev=elev, azim=azim)
        axes_pred.append(ax)
        last_faces_field = faces  # keep last for field colorbar

    # ---- Row 3: Difference (pred - gt) ----
    for ci, t_idx in enumerate(times_idx):
        ax = fig.add_subplot(gs[2, ci], projection="3d")
        ax.set_xlim(0, dims[0])
        ax.set_ylim(0, dims[1])
        ax.set_zlim(0, dims[2])
        try:
            ax.set_box_aspect(dims)
        except Exception:
            pass
        ax.set_axis_off()
        faces = plot_cube_faces(
            (pred - gt)[0, t_idx, :, :, :],
            ax,
            dims=dims,
            contour_levels=50,
            show_back_faces=False,
            vmin=dvmin,
            vmax=dvmax,
            cmap=cmap_diff,
        )
        ax.view_init(elev=elev, azim=azim)
        axes_diff.append(ax)
        last_faces_diff = faces  # for diff colorbar

    # ---- Row labels on the left ----
    if axes_gt:
        axes_gt[0].annotate(
            "Ground Truth",
            xy=(-0.05, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="right",
            fontsize=9,
            rotation=90,
        )
    if axes_pred:
        axes_pred[0].annotate(
            "Prediction",
            xy=(-0.05, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="right",
            fontsize=9,
            rotation=90,
        )
    if axes_diff:
        axes_diff[0].annotate(
            "Difference",
            xy=(-0.05, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="right",
            fontsize=9,
            rotation=90,
        )

    # ---- Shared colorbars on the right ----
    # Field colorbar aligned with GT and Pred rows (top two rows), but shorter vertically
    cax_field = fig.add_axes([0.89, 0.49, 0.015, 0.24])  # [left, bottom, width, height]
    mappable_field = last_faces_field[0]
    fig.colorbar(
        mappable_field, cax=cax_field, orientation="vertical", ticks=[0.0, 0.5, 1]
    )

    # Diff colorbar aligned with Diff row (bottom row), but shorter vertically
    cax_diff = fig.add_axes([0.89, 0.2, 0.015, 0.12])  # [left, bottom, width, height]
    mappable_diff = last_faces_diff[0]
    mappable_diff.set_clim(-0.4, 0.4)
    fig.colorbar(
        mappable_diff, cax=cax_diff, orientation="vertical", ticks=[-0.4, 0, 0.4]
    )

    # ---- Save ----
    os.makedirs(anim_dir, exist_ok=True)
    out_path = os.path.join(anim_dir, f"paper_panel_{index}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def animation_3d(
    gt,
    pred,
    anim_dir: str,
    anim_name: str,
    feature: Literal["T", "u", "v", "w"],
    fps: int = 10,
    rotate: bool = False,
    angle_per_sec: int = 10,
    contour_levels: int = 50,
):
    """data as [channel, time, height, width, depth]"""

    channel = ["T", "u", "v", "w"].index(feature)
    diff = pred - gt

    if channel == 0:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = -1, 1

    frame_idx = 0
    time_length = gt.shape[1]

    fig = plt.figure(figsize=(6, 2), dpi=200)
    ax1 = plt.subplot(1, 3, 1, projection="3d")
    (plt.axis("off"),)
    ax2 = plt.subplot(1, 3, 2, projection="3d")
    plt.axis("off")
    ax3 = plt.subplot(1, 3, 3, projection="3d")
    plt.axis("off")

    orig_faces = plot_cube_faces(
        gt[channel, 0, :, :, :],
        ax1,
        contour_levels=contour_levels,
        show_back_faces=rotate,
        vmin=vmin,
        vmax=vmax,
    )
    pred_faces = plot_cube_faces(
        pred[channel, 0, :, :, :],
        ax2,
        contour_levels=contour_levels,
        show_back_faces=rotate,
        vmin=vmin,
        vmax=vmax,
    )
    diff_faces = plot_cube_faces(
        diff[channel, 0, :, :, :],
        ax3,
        cmap="RdBu_r",
        contour_levels=contour_levels,
        show_back_faces=rotate,
        vmin=-1,
        vmax=1,
    )

    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")
    ax3.set_aspect("equal", adjustable="box")

    ax1.set_title("input")
    ax2.set_title("output")
    ax3.set_title("difference")

    bottom_text = fig.text(0.5, 0.02, "Simulation t = 0", ha="center", va="bottom")

    elev = 15
    ax1.view_init(elev=elev)
    ax2.view_init(elev=elev)
    ax3.view_init(elev=elev)

    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = fig.colorbar(diff_faces[1], cax=cax, ticks=[0, 0.5, 1], extend="both", orientation='vertical')

    last_frame = 0

    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal gt, pred, diff, frame_idx, last_frame
        nonlocal orig_faces, pred_faces, diff_faces
        if frame <= last_frame:
            # no new frame
            return orig_faces + pred_faces + diff_faces

        frame_idx += 1
        last_frame = frame_idx

        # update frames
        reset_contours(orig_faces + pred_faces + diff_faces)
        orig_faces = plot_cube_faces(
            gt[channel, frame_idx, :, :, :],
            ax1,
            contour_levels=contour_levels,
            show_back_faces=rotate,
            vmin=vmin,
            vmax=vmax,
        )
        pred_faces = plot_cube_faces(
            pred[channel, frame_idx, :, :, :],
            ax2,
            contour_levels=contour_levels,
            show_back_faces=rotate,
            vmin=vmin,
            vmax=vmax,
        )
        diff_faces = plot_cube_faces(
            diff[channel, frame_idx, :, :, :],
            ax3,
            cmap="RdBu_r",
            contour_levels=contour_levels,
            show_back_faces=rotate,
            vmin=-1,
            vmax=1,
        )

        # cbar.remove()
        # cbar = fig.colorbar(diff_faces[1], ax=ax3, orientation='vertical')

        bottom_text.set_text(f"Simulation t = {frame_idx}")

        # update color map limits
        set_clims(orig_faces + pred_faces, vmin, vmax)
        set_clims(diff_faces, vmin=-1, vmax=1)

        if rotate:
            azim = -50 + angle_per_sec * frame / fps
            ax1.view_init(elev=elev, azim=azim)
            ax2.view_init(elev=elev, azim=azim)
            ax3.view_init(elev=elev, azim=azim)

        return orig_faces + pred_faces + diff_faces

    # create matplotlib animation
    anim = animation.FuncAnimation(
        fig,
        frame_updater,
        frames=time_length,
        interval=1000 / fps,
        blit=True,
    )
    os.makedirs(anim_dir, exist_ok=True)
    out = os.path.join(anim_dir, anim_name)
    anim.save(out, dpi=500)

    plt.close()
    return out


def plot_cube_faces(
    arr,
    ax,
    dims=(2 * np.pi, 2 * np.pi, 2),
    contour_levels=100,
    cmap="rainbow",
    show_back_faces=False,
    vmin=None,
    vmax=None,
    **contour_kwargs,
) -> list:
    # transpose back to julia order to reuse code TODO
    arr = np.transpose(arr, (2, 0, 1))

    z0 = np.linspace(0, dims[2], arr.shape[2])
    x0 = np.linspace(0, dims[0], arr.shape[0])
    y0 = np.linspace(0, dims[1], arr.shape[1])
    x, y, z = np.meshgrid(x0, y0, z0)

    xmax, ymax, zmax = max(x0), max(y0), max(z0)
    _vmin, _vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if vmin is None:
        vmin = _vmin
    if vmax is None:
        vmax = _vmax
    levels = np.linspace(vmin, vmax, contour_levels)

    z1 = ax.contourf(
        x[:, :, 0],
        y[:, :, 0],
        arr[:, :, -1].T,
        zdir="z",
        offset=zmax,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        **contour_kwargs,
    )
    if show_back_faces:
        z2 = ax.contourf(
            x[:, :, 0],
            y[:, :, 0],
            arr[:, :, 0].T,
            zdir="z",
            offset=0,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
        )
    y1 = ax.contourf(
        x[0, :, :].T,
        arr[:, 0, :].T,
        z[0, :, :].T,
        zdir="y",
        offset=0,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        **contour_kwargs,
    )
    if show_back_faces:
        y2 = ax.contourf(
            x[0, :, :].T,
            arr[:, -1, :].T,
            z[0, :, :].T,
            zdir="y",
            offset=ymax,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
            **contour_kwargs,
        )
    x1 = ax.contourf(
        arr[-1, :, :].T,
        y[:, 0, :].T,
        z[:, 0, :].T,
        zdir="x",
        offset=xmax,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        **contour_kwargs,
    )
    if show_back_faces:
        x2 = ax.contourf(
            arr[0, :, :].T,
            y[:, 0, :].T,
            z[:, 0, :].T,
            zdir="x",
            offset=0,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
            **contour_kwargs,
        )

    return [z1, z2, y1, y2, x1, x2] if show_back_faces else [z1, y1, x1]


def reset_contours(contours):
    for cont in contours:
        cont.remove()


def set_clims(contours, vmin, vmax):
    for cont in contours:
        cont.set_clim(vmin, vmax)

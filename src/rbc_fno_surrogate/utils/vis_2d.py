from abc import ABC, abstractmethod
import pathlib
import tempfile
import time
from typing import Any, List, Tuple

import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)


class Visualizer(ABC):
    def __init__(
        self,
        size: List[int],
        cmap: str,
        display: bool,
        fps: int = 10,
    ) -> None:
        self.size = size
        self.H, self.W = size
        self.cmap = cmap
        self.show = display
        self.setup = False
        self.fps = fps

    def _setup_display(self) -> None:
        if self.show:
            # Do not raise/steal focus on macOS when figures draw
            mpl.rcParams["figure.raise_window"] = False
            plt.ion()
            plt.show()
        else:
            matplotlib.use("Agg")

        self.setup = True

    def display(self) -> None:
        # setup display
        if not self.setup:
            self._setup_display()

        # control framerate
        dt = 1.0 / self.fps if self.fps > 0 else 0.0
        t0 = time.time()

        # redraw all open figures
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # let GUI update briefly
            elapsed = time.time() - t0
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

    @abstractmethod
    def update(self, data: Any) -> Figure:
        pass


class TemperatureVisualizer(Visualizer):
    def __init__(
        self,
        size: List[int] = [64, 96],
        vmin: float = None,
        vmax: float = None,
        display: bool = False,
        fps: int = 10,
    ) -> None:
        # params
        super().__init__(size=size, cmap="coolwarm", display=display, fps=fps)
        self.vmin = vmin
        self.vmax = vmax

        # Temperature field
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow(
            np.zeros(self.size),
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
        )

        # Axes settings
        self.ax.set_axis_off()

    def update(self, data: Any) -> Figure:
        self.image.set_array(data)
        self.display()
        return self.fig


class PredictionVisualizer(Visualizer):
    def __init__(
        self,
        size: List[int] = [64, 96],
        field: str = "T",
        display: bool = False,
        fps: int = 10,
    ) -> None:
        # params
        super().__init__(size=size, cmap="rainbow", display=display, fps=fps)

        # default vmin/vmax depending on field
        if field == "T":
            self.vmin, self.vmax = 1.0, 2.75
        else:
            self.vmin, self.vmax = -1, 1

        # Create figure with three subplots
        self.fig, self.axes = plt.subplots(
            1, 3, figsize=(9, 3), constrained_layout=True
        )

        # Ground truth and prediction share same normalization and colorbar
        zeros = np.zeros(self.size)
        self.im_gt = self.axes[0].imshow(
            zeros,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
        )
        self.im_pred = self.axes[1].imshow(
            zeros,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
        )

        # Difference plot; no separate colorbar (only one overall colorbar)
        self.im_diff = self.axes[2].imshow(
            zeros,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            origin="lower",
        )

        # Axes titles
        self.axes[0].set_title("Ground Truth")
        self.axes[1].set_title("Prediction")
        self.axes[2].set_title("Difference")

        # Turn off ticks/axes
        for ax in self.axes:
            ax.set_axis_off()

        # Single colorbar for the main field (GT/Prediction)
        # self.cbar = self.fig.colorbar(
        #     self.im_gt, ax=self.axes[:2], fraction=0.04, pad=0.015, shrink=0.65
        # )

    def update(self, data: Tuple[Any]) -> Figure:
        """
        Update the visualization.

        Parameters
        ----------
        data : tuple
            (target, prediction) or (target, prediction, time_value)
            where each frame has shape matching `size`.
        """
        if len(data) == 3:
            target, pred, t_value = data
            self.fig.suptitle(f"t = {t_value}", fontsize=12)
        else:
            target, pred = data
            self.fig.suptitle("")

        # Update image data
        diff = pred - target
        self.im_gt.set_array(target)
        self.im_pred.set_array(pred)
        self.im_diff.set_array(diff)

        # Redraw
        self.display()
        return self.fig


def sequence2video(
    target,
    prediction,
    field: str = "T",
    fps: int = 2,
) -> str:
    # set up path
    path = pathlib.Path(f"{tempfile.gettempdir()}/rbcfno").resolve()
    path.mkdir(parents=True, exist_ok=True)

    # map field to channel index
    if field == "T":
        channel = 0
    elif field == "U":
        channel = 1
    elif field == "W":
        channel = 2
    else:
        raise ValueError(f"Unknown field: {field}")

    # shapes: (C, T, H, W)
    _, steps, H, W = target.shape

    # set up prediction visualizer (single colorbar for GT/pred)
    vis = PredictionVisualizer(size=[H, W], field=field, display=False, fps=fps)
    fig = vis.fig

    def animate(i: int):
        target_frame = target[channel, i]
        pred_frame = prediction[channel, i]
        vis.update((target_frame, pred_frame, i))
        return [vis.im_gt, vis.im_pred, vis.im_diff]

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=steps,
        blit=True,
    )

    # save as mp4
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    path = path / f"video_{field}.mp4"
    ani.save(path, writer=writer)
    plt.close(fig)
    return str(path)

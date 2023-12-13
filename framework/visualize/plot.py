from re import L
import torch
import os
import numpy as np
from framework.utils import port
from ..utils import U
from typing import Dict, Tuple, List, Optional, Callable, Union, Iterable, Any
import threading
import atexit
from torch.multiprocessing import Process, Queue, Event
from queue import Empty as EmptyQueue
from queue import Full as FullQueue
import sys
import itertools
import PIL
import time
import scipy.cluster.hierarchy as spc

wandb = None
plt = None
make_axes_locatable = None
FigureCanvas = None


def make_dict(keys: Iterable[str], a: Any) -> Dict[str, Any]:
    if not isinstance(a, dict):
        a = {k: a for k in keys}
    return a


def import_matplotlib():
    global plt
    global make_axes_locatable
    global FigureCanvas
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.backends.backend_agg import FigureCanvas


class CustomPlot:
    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        pass

    def to_wandb(self):
        return None


class Text(CustomPlot):
    def __init__(self, text: str):
        self.text = text

    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        summary_writer.add_text(name, self.text, global_step)

    def to_wandb(self):
        return wandb.Html(self.text.replace("\n", "<br>"))


class Histogram(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], n_bins: int = 64):
        if torch.is_tensor(data):
            data = data.detach().cpu()

        self.data = data
        self.n_bins = n_bins

    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        summary_writer.add_histogram(name, self.data, global_step, max_bins=self.n_bins)

    def to_wandb(self):
        return wandb.Histogram(self.data, num_bins=self.n_bins)


class Image(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], caption: Optional[str] = None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        self.data = data.astype(np.float32)
        self.caption = caption

    def to_tensorboard(self, name, summary_writer, global_step):
        if self.data.shape[-1] in [1,3]:
            data = np.transpose(self.data, (2, 0, 1))
        else:
            data = self.data
        summary_writer.add_image(name, data, global_step)

    def to_wandb(self):
        if self.data.shape[0] in [1, 3]:
            data = np.transpose(self.data, (1, 2, 0))
        else:
            data = self.data

        if data.ndim == 3 and data.shape[-1] == 1:
            mode = "L"
            data = data[..., 0]
        else:
            mode = "RGB"

        data = PIL.Image.fromarray(np.ascontiguousarray((data * 255.0).astype(np.uint8)), mode=mode)
        return wandb.Image(data, caption=self.caption)


class Scalars(CustomPlot):
    def __init__(self, scalar_dict: Dict[str, Union[torch.Tensor, np.ndarray, int, float]]):
        self.values = {k: v.item() if torch.is_tensor(v) else v for k, v in scalar_dict.items()}
        self.leged = sorted(self.values.keys())

    def to_tensorboard(self, name, summary_writer, global_step):
        v = {k: v for k, v in self.values.items() if v == v}
        summary_writer.add_scalars(name, v, global_step)

    def to_wandb(self):
        return self.values


class Scalar(CustomPlot):
    def __init__(self, val: Union[torch.Tensor, np.ndarray, int, float]):
        if torch.is_tensor(val):
            val = val.item()

        self.val = val

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_scalar(name, self.val, global_step)

    def to_wandb(self):
        return self.val


class MatplotlibPlot(CustomPlot):
    def __init__(self, as_image: bool = False):
        import_matplotlib()
        self.as_image = as_image

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_figure(name, self.matplotlib_plot(), global_step)

    def to_wandb(self):
        plot = self.matplotlib_plot()

        if self.as_image:
            plot = FigureCanvas(plot)
            plot.draw()
            plot = PIL.Image.frombuffer('RGBA', plot.get_width_height(), plot.buffer_rgba(), 'raw', 'RGBA', 0, 1)
            plot = wandb.Image(plot)

        return plot


class ImageGrid(MatplotlibPlot):
    def __init__(self, images: List[Union[torch.Tensor, np.ndarray]], grid_size: Tuple[int, int], figsize=None):
        super().__init__()

        def convert_image(i):
            return i.detach().cpu().numpy()

        self.images = U.apply_to_tensors(images, convert_image)
        self.grid_size = grid_size
        self.figsize = figsize

    def matplotlib_plot(self):
        figs = [self.images[i * self.grid_size[1] : (i + 1) * self.grid_size[1]] for i in range(self.grid_size[0])]
        figsize = self.figsize
        if figsize is None:
            h = sum(figs[i][0].shape[0] for i in range(self.grid_size[0]))
            w = sum(figs[0][i].shape[1] for i in range(self.grid_size[1]))
            figsize = (w/30, h/30)

        fig, ax = plt.subplots(self.grid_size[0], self.grid_size[1], figsize=figsize, squeeze=False)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                plt.sca(ax[i, j])
                plt.imshow(figs[i][j])
                plt.yticks([])
                plt.xticks([])
        return fig


class Barplot(MatplotlibPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], names: Optional[List[str]] = None,
                 stds: Optional[Union[torch.Tensor, np.ndarray]] = None, logy: bool = False, tick_rotation: float = 0,
                 font_size: int = 8, xlabel: Optional[str] = None, ylabel: Optional[str] = None, width: Optional[float] = None):
        # Always render as image if names given because of a W&B bug that makes xticks disappear
        super().__init__(as_image=names is not None)

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        if torch.is_tensor(stds):
            stds = stds.detach().cpu().numpy()

        assert data.ndim == 1
        assert (names is None) or (data.shape[0] == len(names))

        self.data = data.tolist()
        self.names = names
        self.stds = stds
        self.logy = logy
        self.tick_rotation = tick_rotation
        self.font_size = font_size
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.width = width

    def matplotlib_plot(self):
        f = plt.figure()

        x = [i for i in range(len(self.data))]
        extra_args = {}
        if self.width is not None:
            extra_args["width"] = self.width
        plt.bar(x, self.data, yerr=self.stds, **extra_args)
        if self.names is not None:
            extra_args = dict(ha="right", rotation_mode="anchor", rotation=self.tick_rotation) if (self.tick_rotation != 0) else {}
            plt.xticks(x, self.names, fontsize=self.font_size, **extra_args)

        if self.logy:
            plt.yscale("log")

        if self.xlabel:
            plt.xlabel(self.xlabel)

        if self.ylabel:
            plt.ylabel(self.ylabel)

        return f


class Barplots(MatplotlibPlot):
    def __init__(self, data: Dict[str, Union[torch.Tensor, np.ndarray]], names: Dict[str, Optional[List[str]]] = {},
                offsets: Dict[str, int] = {}):
        super().__init__()

        data = {
            k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in data.items()
        }

        for d in data.values():
            assert d.ndim == 1

        self.data = {
            k: v.tolist() for k, v in data.items()
        }
        self.names = names
        self.offsets = offsets

    def matplotlib_plot(self):
        f = plt.figure()

        names = list(sorted(self.data.keys()))

        for name in names:
            o = self.offsets.get(name)
            plt.bar([i for i in range(o, o+len(self.data[name]))], self.data[name])
            plt.xticks(self.names.get(name))

        plt.legend(names)
        return f


class XYChart(MatplotlibPlot):
    def __init__(self, data: Dict[str, List[Tuple[float, float]]], markers: List[Tuple[float, float]] = [],
                 xlim=(None, None), ylim=(None, None), point_markers=None, line_styles=None, point_marker_size=None,
                 alpha=None, legend=True):
        super().__init__()

        self.data = U.apply_to_tensors(data, lambda x: x.item())
        self.xlim = xlim
        self.ylim = ylim
        self.markers = markers
        self.point_markers = make_dict(data.keys(), point_markers)
        self.point_marker_size = make_dict(data.keys(), point_marker_size)
        self.line_styles = make_dict(data.keys(), line_styles)
        self.legend = legend
        self.alpha = make_dict(data.keys(), alpha)

    def matplotlib_plot(self):
        f = plt.figure()
        names = list(sorted(self.data.keys()))

        for n in names:
            plt.plot([p[0] for p in self.data[n]], [p[1] for p in self.data[n]], marker=self.point_markers.get(n),
                     linestyle=self.line_styles.get(n), alpha=self.alpha.get(n, 1),
                     markersize=self.point_marker_size.get(n))

        if self.markers:
            plt.plot([a[0] for a in self.markers], [a[1] for a in self.markers], linestyle='', marker='o',
                 markersize=2, zorder=999999)

        if self.legend:
            plt.legend(names)
        plt.ylim(*self.xlim)
        plt.xlim(*self.ylim)

        return f


class Heatmap(MatplotlibPlot):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                 round_decimals: Optional[int] = None, x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None, textval: bool = True, subsample_ticks: int = 1,
                 cmap="auto", colorbar: bool = True, range=(None, None), figsize=None, ticksize: int = 8):

        super().__init__()
        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.round_decimals = round_decimals
        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks
        self.textval = textval
        self.subsample_ticks = subsample_ticks
        self.cmap = plt.cm.Blues if cmap=="auto" else cmap
        self.colorbar = colorbar
        self.range = range
        self.figsize = figsize
        self.ticksize = ticksize

    def get_marks(self, m: Optional[Union[str, List[str]]], n: int):
        if not m:
            return m

        assert len(m) == n
        return [l for i, l in enumerate(m) if i % self.subsample_ticks == 0]

    def plot_extra(self, figure):
        pass

    def raw_state(self) -> Dict:
        return {k: self.__dict__[k] for k in {"map", "x_marks", "y_marks", "xlabel", "ylabel"}}

    def matplotlib_plot(self):
        figure, ax = plt.subplots(figsize=self.figsize)
        figure.set_tight_layout(True)

        im = plt.imshow(self.map.astype(np.float32), interpolation='nearest', cmap=self.cmap, aspect='auto', vmin=self.range[0], vmax=self.range[1])

        x_marks = self.get_marks(self.x_marks, self.map.shape[1])
        y_marks = self.get_marks(self.y_marks, self.map.shape[0])

        if x_marks is not None:
            if not x_marks:
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                plt.xticks(np.arange(self.map.shape[1])[::self.subsample_ticks], x_marks, rotation=45,
                           fontsize=self.ticksize, ha="right", rotation_mode="anchor")

        if y_marks is not None:
            if not y_marks:
                plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
            else:
                plt.yticks(np.arange(self.map.shape[0])[::self.subsample_ticks], y_marks, fontsize=self.ticksize)

        if self.textval:
            # Use white text if squares are dark; otherwise black.
            threshold = (self.map.max() + self.map.min()) / 2.

            rmap = np.around(self.map, decimals=self.round_decimals) if self.round_decimals is not None else self.map
            for i, j in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
                color = "white" if self.map[i, j] > threshold else "black"
                plt.text(j, i, rmap[i, j], ha="center", va="center", color=color, fontsize=8)

        self.plot_extra(figure)

        if self.ylabel:
            plt.ylabel(self.ylabel)
        if self.xlabel:
            plt.xlabel(self.xlabel)

        if self.colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=0.1)
            plt.colorbar(im, cax)

        return figure


class Clustermap(Heatmap):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], label: Optional[str] = None,
                 round_decimals: Optional[int] = None, marks: Optional[List[str]] = None,
                 xmarks: bool = True, show_clusters: bool = True, **kwargs):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        assert map.ndim == 2
        assert map.shape[0] == map.shape[1]

        pdist = spc.distance.pdist(map)
        linkage = spc.linkage(pdist, method='complete')

        self.orig_map = map
        self.idx_to_cluster = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
        self.idx = np.lexsort((np.arange(self.idx_to_cluster.shape[0]), self.idx_to_cluster))

        map = map[self.idx, :][:, self.idx]

        if marks is not None:
            marks = [marks[i] for i in self.idx]

        self.show_clusters = show_clusters

        super().__init__(map=map, xlabel=label, ylabel=label, round_decimals=round_decimals,
                         x_marks=marks if xmarks else [], y_marks=marks, **kwargs)

    def raw_state(self) -> Dict:
        res = super().raw_state()
        res.update({k: self.__dict__[k] for k in {"orig_map", "idx_to_cluster"}})
        return res

    def plot_extra(self, fig):
        if self.show_clusters:
            n_clusters = self.idx_to_cluster.max()
            cluster_offsets = np.cumsum([(self.idx_to_cluster == (c+1)).sum() for c in range(n_clusters)]).tolist()
            cluster_offsets = [0] + cluster_offsets

            for i in range(1, len(cluster_offsets)):
                # bottom
                plt.plot([cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], [cluster_offsets[i]-0.5, cluster_offsets[i]-0.5], color="white", linewidth=1)

                # top
                plt.plot([cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], [cluster_offsets[i-1]-0.5, cluster_offsets[i-1]-0.5], color="white", linewidth=1)

                #right
                plt.plot([cluster_offsets[i]-0.5, cluster_offsets[i]-0.5], [cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], color="white", linewidth=1)

                #left
                plt.plot([cluster_offsets[i-1]-0.5, cluster_offsets[i-1]-0.5], [cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], color="white", linewidth=1)


class AnimatedHeatmap(CustomPlot):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], xlabel: str, ylabel: str,
                 round_decimals: Optional[int] = None, x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None, textval: bool = True, subsample_ticks:int = 1,
                 fps: float = 2, cmap = "auto", colorbar: bool = True, colorbar_ticks = None,
                 colorbar_labels = None, ignore_wrong_marks: bool = False):

        super().__init__()
        import_matplotlib()

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.round_decimals = round_decimals
        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks
        self.textval = textval
        self.subsample_ticks = subsample_ticks
        self.fps = fps
        self.cmap = plt.cm.Blues if cmap=="auto" else cmap
        self.colorbar = colorbar
        self.colorbar_ticks = colorbar_ticks
        self.colorbar_labels = colorbar_labels
        self.ignore_wrong_marks = ignore_wrong_marks

        assert (colorbar_labels is None) == (colorbar_ticks is None)

    def get_marks(self, m: Optional[Union[str, List[str]]], n: int):
        if m is None:
            return None

        if self.ignore_wrong_marks and len(m) != n:
            return None

        assert len(m) == n
        return [l for i, l in enumerate(m) if i % self.subsample_ticks == 0]

    def to_video(self):
        data = self.map.astype(np.float32)

        x_marks = self.get_marks(self.x_marks, self.map.shape[2])
        y_marks = self.get_marks(self.y_marks, self.map.shape[1])

        figure, ax = plt.subplots()
        canvas = FigureCanvas(figure)
        figure.set_tight_layout(True)

        im = plt.imshow(data[0], interpolation='nearest', cmap=self.cmap, aspect='auto', animated=True,
                        vmin = data.min(), vmax=data.max())

        if x_marks is not None:
            plt.xticks(np.arange(self.map.shape[2])[::self.subsample_ticks], x_marks, rotation=45, fontsize=8,
                    ha="right", rotation_mode="anchor")

        if y_marks is not None:
            plt.yticks(np.arange(self.map.shape[1])[::self.subsample_ticks], y_marks, fontsize=8)

        title = plt.title("Step: 0")

        if self.textval:
            # Use white text if squares are dark; otherwise black.
            threshold = self.map.max() / 2.

            rmap = np.around(self.map, decimals=self.round_decimals) if self.round_decimals is not None else self.map
            for i, j in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
                color = "white" if self.map[i, j] > threshold else "black"
                plt.text(j, i, rmap[i, j], ha="center", va="center", color=color, fontsize=8)

        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        if self.colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=0.1)
            cbar = plt.colorbar(im, cax, ticks=self.colorbar_ticks)

            if self.colorbar_labels is not None:
                cbar.ax.set_yticklabels(self.colorbar_labels)


        frames = []
        for i in range(data.shape[0]):
            canvas.draw()
            image_from_plot = np.array(canvas.renderer.buffer_rgba())
            frames.append(image_from_plot.reshape(figure.canvas.get_width_height()[::-1] + (4,))[:,:,:3])

            if i < data.shape[0] - 1:
                im.set_data(data[i + 1])
                title.set_text(f"Step: {i + 1}")

        del figure

        video = np.stack(frames, 0)
        return np.transpose(video, (0, 3, 1, 2))

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_video(name, self.to_video()[np.newaxis], global_step, fps = self.fps)

    def to_wandb(self):
        return wandb.Video(self.to_video(), fps = self.fps)


class ConfusionMatrix(Heatmap):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], class_names: Optional[List[str]] = None,
                 x_marks: Optional[List[str]] = None, y_marks: Optional[List[str]] = None):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        map = np.transpose(map, (1, 0))
        map = map.astype('float') / map.sum(axis=1).clip(1e-6, None)[:, np.newaxis]

        if class_names is not None:
            assert x_marks is None and y_marks is None
            x_marks = y_marks = class_names

        super().__init__(map, "predicted", "real", round_decimals=2, x_marks = x_marks, y_marks = y_marks)


class TextTable(CustomPlot):
    def __init__(self, header: List[str], data: List[List[str]]):
        self.header = header
        self.data = data

    def to_markdown(self):
        res = " | ".join(self.header)+"\n"
        res += " | ".join("---" for _ in self.header)+"\n"
        return res+"\n".join([" | ".join([x.replace("|", "&#124;") for x in l]) for l in self.data])

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_text(name, self.to_markdown(), global_step)

    def to_wandb(self):
        return wandb.Table(data=self.data, columns=self.header)


class PlotAsync:
    @staticmethod
    def worker(self, fn, *args):
        try:
            self.result = fn(*args)
        except:
            self.failed = True
            raise

    def __init__(self, fn: Callable[[any], Dict[str, any]], args: Tuple=()):
        self.result = None
        self.failed = False

        args = U.apply_to_tensors(args, lambda x: x.detach().cpu().clone())

        self.thread = threading.Thread(target = self.worker, args=(self, fn, *args), daemon=True)
        self.thread.start()

    def get(self, wait: bool) -> Optional[Dict[str, any]]:
        if (self.result is None and not wait) or self.failed:
            return None

        self.thread.join()
        return self.result


class Logger:
    @staticmethod
    def parse_switch_string(s: str) -> Tuple[bool,bool]:
        s = s.lower()
        if s=="all":
            return True, True
        elif s=="none":
            return False, False

        use_tb, use_wandb =  False, False
        s = s.split(",")
        for p in s:
            if p=="tb":
                use_tb = True
            elif p=="wandb":
                use_wandb = True
            else:
                assert False, "Invalid visualization switch: %s" % p

        return use_tb, use_wandb

    def create_loggers(self):
        self.is_sweep = False
        self.wandb_id = {}
        global wandb

        if self.use_wandb:
            import wandb
            wandb.init(**self.wandb_init_args)
            self.wandb_id = {
                "sweep_id": wandb.run.sweep_id,
                "run_id": wandb.run.id,
                "project": wandb.run.project
            }
            self.is_sweep = bool(wandb.run.sweep_id)
            wandb.config["is_sweep"] = self.is_sweep
            wandb.config.update(self.wandb_extra_config, allow_val_change=True)

            self.save_dir = os.path.join(wandb.run.dir)

        os.makedirs(self.save_dir, exist_ok=True)
        self.tb_logdir = os.path.join(self.save_dir, "tensorboard")

        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.tb_logdir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=self.tb_logdir, flush_secs=30)
        else:
            self.summary_writer = None

    def __init__(self, save_dir: Optional[str] = None, use_tb: bool = False, use_wandb: bool = False,
                 get_global_step: Optional[Callable[[], int]] = None, wandb_init_args={}, wandb_extra_config={}):
        global plt
        global wandb

        import_matplotlib()

        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.save_dir = save_dir
        self.get_global_step = get_global_step
        self.wandb_init_args = wandb_init_args
        self.wandb_extra_config = wandb_extra_config

        self.create_loggers()

    def flatten_dict(self, dict_of_elems: Dict) -> Dict:
        res = {}
        for k, v in dict_of_elems.items():
            if isinstance(v, dict):
                v = self.flatten_dict(v)
                for k2, v2 in v.items():
                    res[k+"/"+k2] = v2
            else:
                res[k] = v
        return res

    def get_step(self, step: Optional[int] = None) -> Optional[int]:
        if step is None and self.get_global_step is not None:
            step = self.get_global_step()

        return step

    def log(self, plotlist: Union[List, Dict, PlotAsync], step: Optional[int] = None):
        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p.get(True) if isinstance(p, PlotAsync) else p for p in plotlist if p]
        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        d = {}
        for p in plotlist:
            d.update(p)

        self.log_dict(d, step)

    def log_dict(self, dict_of_elems: Dict, step: Optional[int] = None):
        dict_of_elems = self.flatten_dict(dict_of_elems)

        if not dict_of_elems:
            return

        dict_of_elems = {k: v.item() if torch.is_tensor(v) and v.nelement()==1 else v for k, v in dict_of_elems.items()}
        dict_of_elems = {k: Scalar(v) if isinstance(v, (int, float)) else v for k, v in dict_of_elems.items()}

        step = self.get_step(step)

        if self.use_wandb:
            wandbdict = {}
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v = v.to_wandb()
                    if v is None:
                        continue

                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            wandbdict[k+"/"+k2] = v2
                    else:
                        wandbdict[k] = v
                elif isinstance(v, plt.Figure):
                    wandbdict[k] = v
                else:
                    assert False, f"Invalid data type {type(v)} for key {k}"

            wandbdict["iteration"] = step
            wandb.log(wandbdict)

        if self.summary_writer is not None:
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v.to_tensorboard(k, self.summary_writer, step)
                elif isinstance(v, plt.Figure):
                    self.summary_writer.add_figure(k, v, step)
                else:
                    assert False, f"Unsupported type {type(v)} for entry {k}"

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def flush(self):
        pass

    def finish(self):
        pass


class AsyncLogger(Logger):
    def redirect(self, f, tag):
        old_write = f.write
        old_writelines = f.writelines
        def new_write(text):
            old_write(text)
            self.print_queue.put((tag, text))
        def new_writelines(lines):
            old_writelines(lines)
            self.print_queue.put((tag, os.linesep.join(lines)))
        f.write = new_write
        f.writelines = new_writelines
        return f

    def wandb_flush_io(self):
        if not self.use_wandb:
            pass

        while not self.print_queue.empty():
            tag, text = self.print_queue.get()
            # wandb.run._redirect_cb("stdout", text)
            wandb.run._console_callback("stdout", text)
            # wandb.run._redirect_cb("stderr" if tag==1 else "stdout", text)

    @staticmethod
    def log_fn(self, stop_event: Event):
        try:
            self._super_create_loggers()
            self.resposne_queue.put({k: self.__dict__[k] for k in ["save_dir", "tb_logdir", "is_sweep", "wandb_id"]})

            while True:
                self.wandb_flush_io()

                try:
                    cmd = self.draw_queue.get(True, 0.1)
                except EmptyQueue:
                    if stop_event.is_set():
                        break
                    else:
                        continue

                self._super_log(*cmd)
                self.resposne_queue.put(True)
        except:
            print("Logger process crashed.")
            raise
        finally:
            try:
                self.wandb_flush_io()
            except:
                pass

            print("Logger: syncing")
            if self.use_wandb:
                wandb.join()

            stop_event.set()
            print("Logger process terminating...")

    def create_loggers(self):
        self._super_create_loggers = super().create_loggers
        self.stop_event = Event()
        self.stop_requested = False
        self.proc = Process(target=self.log_fn, args=(self, self.stop_event))
        self.proc.start()

        atexit.register(self.finish)

    def __init__(self, *args, queue_size: int = 1000, **kwargs):
        self.queue = []

        self.queue_size = queue_size
        self.print_queue = Queue()
        self.draw_queue = Queue(queue_size)
        self.resposne_queue = Queue()
        self._super_log = super().log
        self.waiting = 0

        super().__init__(*args, **kwargs)

        self.__dict__.update(self.resposne_queue.get(True))

        if self.use_wandb:
            # monkey-patch stdout and stderr such that we can redirect it to wandb running in the other process
            sys.stdout = self.redirect(sys.stdout, 0)
            sys.stderr = self.redirect(sys.stderr, 1)

    def log(self, plotlist, step=None):
        if self.stop_event.is_set() or not self.proc.is_alive():
            assert self.stop_requested, "Logger process crashed, but trying to log"
            return

        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        plotlist = U.apply_to_tensors(plotlist, lambda x: x.detach().cpu())

        if step is None:
            step = self.get_global_step()

        self.queue.append((plotlist, step))
        self.flush(wait = False)

    def enqueue(self, data, step: Optional[int]):
        while True:
            if not self.proc.is_alive():
                return

            try:
                self.draw_queue.put((data, step), timeout=1)
                break
            except TimeoutError:
                pass
            except FullQueue:
                time.sleep(0.1)
                pass

        self.waiting += 1

    def wait_logger(self, wait = False):
        cond = (lambda: not self.resposne_queue.empty()) if not wait else (lambda: self.waiting>0)
        already_printed = False
        while cond() and not self.stop_event.is_set() and self.proc.is_alive():
            will_wait = self.resposne_queue.empty()
            if will_wait and not already_printed:
                already_printed = True
                sys.stdout.write("Warning: waiting for logger... ")
                sys.stdout.flush()
            try:
                self.resposne_queue.get(True, 0.2)
            except EmptyQueue:
                continue
            self.waiting -= 1

        if already_printed:
            print("done.")

    def request_stop(self):
        self.stop_requested = True
        self.stop_event.set()

    def flush(self, wait: bool = True):
        while self.proc.is_alive() and self.queue:
            plotlist, step = self.queue[0]

            for i, p in enumerate(plotlist):
                if isinstance(p, PlotAsync):
                    res = p.get(wait)
                    if res is not None:
                        plotlist[i] = res
                    else:
                        if wait:
                            assert p.failed
                            # Exception in the worker thread
                            print("Exception detected in a PlotAsync object. Syncing logger and ignoring further plots.")
                            self.wait_logger(True)
                            self.request_stop()
                            self.proc.join()

                        return

            self.queue.pop(0)
            self.enqueue(plotlist, step)

        self.wait_logger(wait)

    def finish(self):
        if self.stop_event.is_set():
            return

        self.flush(True)
        self.request_stop()
        self.proc.join()


class CustomHistogram(Barplot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], n_bins: int = 64, ticks: str = "centroids",
                 format: str = "%.2f", logy: bool = False, tick_rotation: float = 45, fontsize: int = 8,
                 xlabel: Optional[str] = None, width: Optional[float] = None, logx: bool = False):

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        if logx:
            m = np.max(data)
            assert np.min(data) >= 0
            bins = 2 ** np.arange(0, np.log2(m))
        else:
            bins = n_bins

        hist, bins = np.histogram(data, bins=bins)
        if ticks=="centroids":
            centroids = (bins[1:] + bins[:-1])/2
            ticks = [format % c for c in centroids]
        elif ticks=="range":
            ticks = [(format % bins[i])+"-"+(format % bins[i+1]) for i in range(n_bins)]
        elif ticks=="right":
            ticks = [format % c for c in bins[1:]]
        else:
            raise ValueError("Invalid value for ticks: %s" % ticks)

        super().__init__(hist, ticks, logy=logy, tick_rotation=tick_rotation, font_size=fontsize, xlabel=xlabel,
                         width=width)

import hydra
import h5py
from omegaconf import DictConfig
from tqdm import tqdm
from rbc_fno_surrogate.utils.vis_2d import TemperatureVisualizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="vis")
def vis(config: DictConfig):
    # params
    path = f"{config['paths']['data_dir']}/{config['path']}"
    episode = config["episode"]

    # Load dataset episode
    with h5py.File(path, "r") as file:
        states = file[f"states{episode}"]

        # data params
        steps = file.attrs.get("steps")
        H, W = states.shape[-2], states.shape[-1]

        # visualizer
        # vis = PredictionVisualizer(size=[H, W], field="T", display=True, fps=10)
        vis = TemperatureVisualizer(size=[H, W], vmin=1.0, vmax=2.0, display=True)

        for step in tqdm(range(steps - 1)):
            state = states[step]

            T = state[0]  # (H, W)

            # Noise for Prediction Visualization Testing
            # rng = np.random.default_rng()
            # sigma = 0.1 * float(T.max() - T.min() + 1e-8)  # ~2% of dynamic range
            # D = (T + rng.normal(0.0, sigma, size=T.shape)).astype(T.dtype, copy=False)

            vis.update(T)


if __name__ == "__main__":
    RA = 10000
    NR = 0
    DATASET = "pd"
    SPLIT = "train"
    print(f"Visualizing RA={RA}, dataset={DATASET}, episode={NR} from {SPLIT} split")

    vis()

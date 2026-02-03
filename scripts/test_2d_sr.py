import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch
from torch import Tensor
from torch.nn import functional as F

from rbc_fno_surrogate.data import RBCDatamodule2D
from rbc_fno_surrogate.model import FNO2DModule
import rbc_fno_surrogate.callbacks.metrics_2d as metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_test")
def main(config: DictConfig):
    # device
    device = best_device()

    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # data
    dm = RBCDatamodule2D(**config["data"])
    dm.setup("test")
    denorm = dm.datasets["test"].denormalize_batch

    # model
    model = FNO2DModule.load_from_checkpoint(config["checkpoint"])
    model.to(device)
    model.eval()

    # wandb run
    wandb.init(
        project=f"RBC-2D-{str(config['model']).capitalize()}",
        config=config,
        dir=output_dir,
        tags=["test", "sr-cons"],
    )

    # loop
    for batch, (x, y) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        # fine and coarse data
        xcoarse = x[:, :, :, ::2, ::2]
        ycoarse = y[:, :, :, ::2, ::2]

        # print shapes
        print(f"x: {x.shape}, xcoarse: {xcoarse.shape}")

        # compute fine prediction
        with torch.no_grad():
            pred_fine = model.predict(x.to(device), y.shape[2]).cpu()
        pred_fine: Tensor = denorm(pred_fine)
        pred_fine_ds = pred_fine[:, :, :, ::2, ::2]

        # compute coarse prediction
        with torch.no_grad():
            pred_coarse = model.predict(xcoarse.to(device), ycoarse.shape[2]).cpu()
        pred_coarse: Tensor = denorm(pred_coarse)

        # 1) Sequence Metrics NRSSE and RMSE
        seq_len = pred_fine.shape[2]
        losses = []
        for t in range(seq_len):
            # metrics per sample and time step
            nrsse = metrics.nrsse(pred_fine_ds[:, :, t], pred_coarse[:, :, t])
            loss = F.mse_loss(pred_fine_ds[:, :, t], pred_coarse[:, :, t])
            losses.append(
                {
                    "batch_idx": batch,
                    "step": t,
                    "loss": loss.item(),
                    "nrsse": nrsse.item(),
                }
            )

    # overall metrics
    df = pd.DataFrame(losses)
    loss = df["loss"].mean()
    nrsse = df["nrsse"].mean()

    # plots
    im = plot_metric(df, "loss")

    wandb.log(
        {
            "test/loss": loss,
            "test/nrsse": nrsse,
            "test/Plot": im,
            "test/Table": wandb.Table(dataframe=df),
        }
    )


def plot_metric(df: pd.DataFrame, metric: str):
    fig = plt.figure()
    sns.set_theme()
    ax = sns.lineplot(data=df, x="step", y=metric)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xlabel("Time Step")
    ax.set_ylim(bottom=0, top=0.8)
    # save as image
    im = wandb.Image(fig, caption=metric)
    plt.close(fig)
    return im


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # first free NVIDIA GPU
    if torch.backends.mps.is_available():  # Apple-silicon
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()

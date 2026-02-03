import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch
from torch import Tensor

from rbc_fno_surrogate.data import RBCDatamodule3D
from rbc_fno_surrogate.model import FNO3DModule, LRAN3DModule, UNet3DModule
from rbc_fno_surrogate.utils.vis_3d import animation_3d, plot_paper
import rbc_fno_surrogate.callbacks.metrics_3d as metrics

HIST_XLIM = (-0.15, 0.2)
HIST_BINS = 100


def pdf_edges_and_centers(xlim=HIST_XLIM, bins=HIST_BINS):
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return edges, centers


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_test")
def main(config: DictConfig):
    # device
    device = best_device()

    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # data
    dm = RBCDatamodule3D(**config["data"])
    dm.setup("test")
    denorm = dm.datasets["test"].denormalize_batch

    # model
    if config["model"] == "fno":
        model = FNO3DModule.load_from_checkpoint(config["checkpoint"])
    elif config["model"] == "lran":
        model = LRAN3DModule.load_from_checkpoint(config["checkpoint"])
    elif config["model"] == "unet":
        model = UNet3DModule.load_from_checkpoint(config["checkpoint"])
    model.to(device)
    model.eval()

    # wandb run
    wandb.init(
        project=f"RBC-3D-{str(config['model']).capitalize()}",
        config=config,
        dir=output_dir,
        tags=config["tags"],
    )

    # loop
    list_metrics = []
    list_nusselt = []
    list_kinetic = []
    list_div = []
    list_profile_q = []
    list_profile_qp = []
    hist_qp = None
    for batch, (x, y) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        # compute predictions and denormalize data
        with torch.no_grad():
            y_hat = model.predict(x.to(device), y.shape[2]).cpu()
        pred: Tensor = denorm(y_hat)
        target: Tensor = denorm(y)

        # 1) Sequence Metrics NRSSE and RMSE
        seq_len = pred.shape[2]
        for t in range(seq_len):
            # metrics per sample and time step
            loss = model.loss(pred[:, :, t], target[:, :, t])
            rmse = metrics.rmse(pred[:, :, t], target[:, :, t])
            nrsse = metrics.nrsse(pred[:, :, t], target[:, :, t])

            list_metrics.append(
                {
                    "batch_idx": batch,
                    "step": t,
                    "rmse": rmse.item(),
                    "nrsse": nrsse.item(),
                    "loss": loss.item(),
                }
            )

        # 2) Visualize samples from first batch element
        path = animation_3d(
            gt=target[0].numpy(),
            pred=pred[0].numpy(),
            feature="T",
            anim_dir=output_dir + "/animations",
            anim_name=f"test_{batch}.mp4",
        )
        plot_paper(
            target[0].numpy(),
            pred[0].numpy(),
            output_dir + "/vis_paper",
            batch,
        )
        video = wandb.Video(path, format="mp4", caption=f"Batch {batch}")
        wandb.log({"test/video": video})

        # 3) Nusselt Number: mean q
        T_mean_ref = target[0, 0].mean()
        for t in range(seq_len):
            nu_pred = metrics.compute_q(pred[0, :, t], T_mean_ref)
            nu_target = metrics.compute_q(target[0, :, t], T_mean_ref)
            list_nusselt.append(
                {
                    "batch_idx": batch,
                    "step": t,
                    "nu_pred": nu_pred.item(),
                    "nu_target": nu_target.item(),
                }
            )

        # 4) Profile of mean q (area-avg over time)
        for t in range(seq_len):
            # q profile
            q_profile_pred = metrics.compute_q(pred[0, :, t], T_mean_ref, profile=True)
            q_profile_target = metrics.compute_q(
                target[0, :, t], T_mean_ref, profile=True
            )
            for z, (p, q) in enumerate(zip(q_profile_pred, q_profile_target)):
                list_profile_q.append(
                    {
                        "batch_idx": batch,
                        "step": t,
                        "height": z,
                        "q_pred": p,
                        "q_target": q,
                    }
                )

        # 5) Profile of q'
        qp_pred = metrics.compute_profile_qprime_rms(pred[0], T_mean_ref)
        qp_target = metrics.compute_profile_qprime_rms(target[0], T_mean_ref)
        for z, (p, q) in enumerate(zip(qp_pred, qp_target)):
            list_profile_qp.append(
                {
                    "batch_idx": batch,
                    "height": z,
                    "qp_pred": p,
                    "qp_target": q,
                }
            )

        # 6) PDF over q' at selected heights
        H = pred[0].shape[3]  # HEIGHT dimension in [C,T,D,H,W]
        zs = [H // 6, H // 2, (5 * H) // 6]
        # lazy-init aggregator once we know H and zs
        if hist_qp is None:
            hist_qp = {
                int(z): {
                    "sum_pred": np.zeros(HIST_BINS, dtype=float),
                    "sum_target": np.zeros(HIST_BINS, dtype=float),
                    "n_items": 0,
                }
                for z in zs
            }

        for z in zs:
            qp_pred_z = metrics.compute_qprime_z(pred[0], T_mean_ref, z)
            qp_targ_z = metrics.compute_qprime_z(target[0], T_mean_ref, z)

            hist_pred, _ = np.histogram(
                qp_pred_z, bins=HIST_BINS, range=HIST_XLIM, density=True
            )
            hist_targ, _ = np.histogram(
                qp_targ_z, bins=HIST_BINS, range=HIST_XLIM, density=True
            )

            hist_qp[int(z)]["sum_pred"] += hist_pred
            hist_qp[int(z)]["sum_target"] += hist_targ
            hist_qp[int(z)]["n_items"] += 1

        # 7) kinetic energy
        for t in range(seq_len):
            kin_pred = metrics.compute_kinetic_energy(pred[0, :, t])
            kin_target = metrics.compute_kinetic_energy(target[0, :, t])
            list_kinetic.append(
                {
                    "batch_idx": batch,
                    "step": t,
                    "kin_pred": kin_pred.item(),
                    "kin_target": kin_target.item(),
                }
            )

        # 8) incompressibility check
        DOMAIN = (4 * np.pi, 4 * np.pi, 2.0)  # (Lx, Ly, Lz)
        for t in range(seq_len):
            # compute divergence using finited diff and fourier
            div_pred = metrics.compute_divergence(pred[0, :, t], DOMAIN)
            div_target = metrics.compute_divergence(target[0, :, t], DOMAIN)

            # RMS values
            pred_rms = torch.sqrt(torch.mean(div_pred**2)).item()
            target_rms = torch.sqrt(torch.mean(div_target**2)).item()

            list_div.append(
                {
                    "batch_idx": batch,
                    "step": t,
                    "div_pred_rms": pred_rms,
                    "div_target_rms": target_rms,
                }
            )

        # 10) reynolds stresses
        pass

    # log metrics
    df_metrics = pd.DataFrame(list_metrics)
    df_nusselt = pd.DataFrame(list_nusselt)
    df_kinetic = pd.DataFrame(list_kinetic)
    df_div = pd.DataFrame(list_div)
    df_profile_q = pd.DataFrame(list_profile_q)
    df_profile_qp = pd.DataFrame(list_profile_qp)

    # aggregated mean PDFs per height
    edges, centers = pdf_edges_and_centers()
    rows = []
    for z, rec in hist_qp.items():
        n = max(1, rec["n_items"])  # guard
        rows.append(
            {
                "height": z,
                "centers": centers,  # store centers for convenience
                "pdf_pred": rec["sum_pred"] / n,
                "pdf_target": rec["sum_target"] / n,
                "n_items": n,
            }
        )
    df_pdf = pd.DataFrame(rows)

    # overall metrics
    rmse = df_metrics["rmse"].mean()
    nrsse = df_metrics["nrsse"].mean()

    # plots
    im_rmse = plot_metric(df_metrics, "rmse")
    im_nrsse = plot_metric(df_metrics, "nrsse")

    wandb.log(
        {
            "test/RMSE": rmse,
            "test/NRSSE": nrsse,
            "test/Table-Metrics": wandb.Table(dataframe=df_metrics),
            "test/Table-Nusselt": wandb.Table(dataframe=df_nusselt),
            "test/Table-Kinetic-Energy": wandb.Table(dataframe=df_kinetic),
            "test/Table-Divergence": wandb.Table(dataframe=df_div),
            "test/Table-Q-Profile": wandb.Table(dataframe=df_profile_q),
            "test/Table-QP-Profile": wandb.Table(dataframe=df_profile_qp),
            "test/Table-QP-Histogram": wandb.Table(dataframe=df_pdf),
            "test/Plot-RMSE": im_rmse,
            "test/Plot-NRSSE": im_nrsse,
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

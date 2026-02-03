import os
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import wandb
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import rbc_fno_surrogate.callbacks.metrics_2d as metrics
from rbc_fno_surrogate.data import RBCDatamodule2D
from rbc_fno_surrogate.model import FNO2DModule, LRAN2DModule, UNet2DModule
from rbc_fno_surrogate.data.dataset import Field2D
from rbc_fno_surrogate.utils.vis_2d import sequence2video
from rbc_fno_surrogate.utils.vis_3d import set_size


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_rbc")
def main(config: DictConfig):
    # device
    device = best_device()

    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # data
    dm = RBCDatamodule2D(**config["data"])
    dm.setup("test")
    denormalize = dm.datasets["test"].denormalize_batch
    DOMAIN = (2 * np.pi, 2.0)  # (Lx, Lz)

    # model
    name = config["model"].pop("name")
    ckpt = config["test"]["checkpoint"]
    if name == "fno":
        model = FNO2DModule.load_from_checkpoint(ckpt)
    elif name == "lran":
        model = LRAN2DModule.load_from_checkpoint(ckpt)
    elif name == "unet":
        model = UNet2DModule.load_from_checkpoint(ckpt)
    else:
        raise ValueError(f"Model {name} not recognized.")
    model.to(device)

    # wandb run
    wandb.init(
        project=f"RBC-2D-{name.upper()}",
        dir=output_dir,
        tags=["test"],
    )

    # loop
    list_metrics = []
    list_profiles = []
    for batch, (x, y) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        with torch.no_grad():
            y_hat = model.predict(x.to(device), y.shape[2]).cpu()
        pred: Tensor = denormalize(y_hat)
        target: Tensor = denormalize(y)

        # 1) Sequence Metrics NRSSE, RMSE, Nusselt, Kinetic Energy, Divergence
        batch_size = pred.shape[0]
        seq_len = pred.shape[2]
        T_mean_ref = target[:, Field2D.T].mean()
        for idx in range(batch_size):
            for t in range(seq_len):
                # metrics
                loss = model.loss(pred[idx, :, t], target[idx, :, t])
                rmse = metrics.rmse(pred[idx, :, t], target[idx, :, t])
                nrsse = metrics.nrsse(pred[idx, :, t], target[idx, :, t])

                # nusselt
                nu_pred = metrics.compute_q(pred[idx, :, t], T_mean_ref)
                nu_target = metrics.compute_q(target[idx, :, t], T_mean_ref)

                # kinetic energy
                kin_pred = metrics.compute_kinetic_energy(pred[idx, :, t])
                kin_target = metrics.compute_kinetic_energy(target[idx, :, t])

                # incompressibility/divergence
                div_pred, _, _ = metrics.compute_divergence(pred[idx, :, t], DOMAIN)
                div_target, _, _ = metrics.compute_divergence(target[idx, :, t], DOMAIN)
                div_pred_rms = torch.sqrt(torch.mean(div_pred**2))
                div_target_rms = torch.sqrt(torch.mean(div_target**2))

                list_metrics.append(
                    {
                        "batch_idx": batch,
                        "sample_idx": idx,
                        "step": t,
                        "rmse": rmse.item(),
                        "nrsse": nrsse.item(),
                        "loss": loss.item(),
                        "nu_pred": nu_pred.item(),
                        "nu_target": nu_target.item(),
                        "kin_pred": kin_pred.item(),
                        "kin_target": kin_target.item(),
                        "div_pred_rms": div_pred_rms.item(),
                        "div_target_rms": div_target_rms.item(),
                    }
                )

        # 2) Compute Profiles of q and qp
        H = pred.shape[3]
        for idx in range(batch_size):
            for t in range(seq_len):
                # q profile
                q_profile_pred = metrics.compute_q(
                    pred[idx, :, t], T_mean_ref, profile=True
                )
                q_profile_target = metrics.compute_q(
                    target[idx, :, t], T_mean_ref, profile=True
                )

                # qp profile
                qp_profile_pred = metrics.compute_profile_qprime_rms(
                    pred[idx], T_mean_ref
                )
                qp_profile_target = metrics.compute_profile_qprime_rms(
                    target[idx], T_mean_ref
                )

                # store
                for z in range(H):
                    list_profiles.append(
                        {
                            "batch_idx": batch,
                            "sample_idx": idx,
                            "step": t,
                            "height": z,
                            "q_pred": q_profile_pred[z],
                            "q_target": q_profile_target[z],
                            "qp_pred": qp_profile_pred[z],
                            "qp_target": qp_profile_target[z],
                        }
                    )

        # 3) State Visualizations
        videos = []
        for field in ["T", "U", "W"]:
            v = sequence2video(target[0], pred[0], field)
            videos.append(wandb.Video(v, caption=field, format="mp4"))
        wandb.log({"test/video": videos})

        # 4) Divergence Visualizations
        div_im = plot_divergence_paper(
            target, pred, DOMAIN, output_dir, index=batch, sample=0, t=0
        )
        wandb.log({"test/divergence": div_im})

    # log metrics
    df_metrics = pd.DataFrame(list_metrics)
    df_profiles = pd.DataFrame(list_profiles)

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
            "test/Table-Profiles": wandb.Table(dataframe=df_profiles),
            "test/Plot-RMSE": im_rmse,
            "test/Plot-NRSSE": im_nrsse,
        }
    )


def plot_divergence_paper(target, pred, domain, out_dir, index=0, sample=0, t=0):
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

    # compute divergence
    div_pred, dudx_pred, dwdz_pred = metrics.compute_divergence(
        pred[sample, :, t], domain
    )
    div_target, dudx_target, dwdz_target = metrics.compute_divergence(
        target[sample, :, t], domain
    )

    # visualize fields and derivatives
    top_fields = [("div", div_target), ("dudx", dudx_target), ("dwdz", dwdz_target)]
    bottom_fields = [("div", div_pred), ("dudx", dudx_pred), ("dwdz", dwdz_pred)]

    # shared color scale across all panels
    all_fields = top_fields + bottom_fields
    stacked = torch.stack([f for _, f in all_fields])
    vmin = stacked.min().item()
    vmax = stacked.max().item()

    # plot
    fig, axes = plt.subplots(
        2, 3, figsize=set_size(455, fraction=1, aspect=0.4), dpi=300
    )
    fig.subplots_adjust(hspace=0.0, wspace=0.15)
    col_titles = [
        r"$\nabla\cdot u$",
        r"$\partial u / \partial x$",
        r"$\partial w / \partial z$",
    ]
    for col, (name, field) in enumerate(top_fields):
        ax = axes[0, col]
        img = ax.imshow(
            field.cpu().detach().numpy(),
            cmap="coolwarm",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(col_titles[col])
        ax.set_xticks([])
        ax.set_yticks([])

    for col, (name, field) in enumerate(bottom_fields):
        ax = axes[1, col]
        img = ax.imshow(
            field.cpu().detach().numpy(),
            cmap="coolwarm",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # row labels to match paper style
    axes[0, 0].set_ylabel("Simulation", labelpad=2)
    axes[1, 0].set_ylabel("Prediction", labelpad=2)

    # shared colorbar
    fig.colorbar(img, ax=axes, shrink=0.7, location="right")

    # save as image
    out_dir = f"{out_dir}/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"divergence_{index}.pdf")
    fig.savefig(out_path, bbox_inches="tight")

    im = wandb.Image(fig, caption="Diveregence and Derivatives")
    plt.close(fig)
    return im


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
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()

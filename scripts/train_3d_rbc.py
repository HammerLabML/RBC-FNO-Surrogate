import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from rbc_fno_surrogate.data import RBCDatamodule3D
from rbc_fno_surrogate.model import (
    FNO3DModule,
    LRAN3DModule,
    UNet3DModule,
)
from rbc_fno_surrogate.callbacks import Metrics3DCallback


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_rbc")
def main(config: DictConfig):
    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # seed
    L.seed_everything(config["seed"], workers=True)

    # data
    dm = RBCDatamodule3D(**config["data"])
    dm.setup("fit")

    # model
    name = config["model"].pop("name")
    if name == "fno":
        model = FNO3DModule(**config["model"])
    elif name == "lran":
        model = LRAN3DModule(**config["model"])
    elif name == "unet":
        model = UNet3DModule(**config["model"])
    else:
        raise ValueError(f"Unknown model name: {name}")

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project=f"RBC-3D-{name.upper()}",
        save_dir=output_dir,
        log_model=False,
        tags="train",
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=8,
        ),
        ModelCheckpoint(
            dirpath=f"{output_dir}/checkpoints/",
            save_top_k=1,
            save_weights_only=True,
            monitor="val/loss",
            mode="min",
        ),
        Metrics3DCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        **config["trainer"],
        logger=logger,
        default_root_dir=output_dir,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")

    # finish logging
    logger.experiment.finish()


if __name__ == "__main__":
    main()

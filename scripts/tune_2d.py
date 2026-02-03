import hydra
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.tuner import Tuner

from rbc_fno_surrogate.data import RBCDatamodule2D
from rbc_fno_surrogate.model import (
    FNO2DModule,
    LRAN2DModule,
    Autoencoder2DModule,
    UNet2DModule,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_rbc")
def main(config: DictConfig):
    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # seed
    L.seed_everything(config["seed"], workers=True)

    # data
    dm = RBCDatamodule2D(**config["data"])
    dm.setup("fit")
    denormalize = dm.datasets["train"].denormalize_batch

    # model
    name = config["model"].pop("name")
    if name == "fno":
        model = FNO2DModule(denormalize=denormalize, **config["model"])
    elif name == "lran":
        model = LRAN2DModule(denormalize=denormalize, **config["model"])
    elif name == "ae":
        model = Autoencoder2DModule(denormalize=denormalize, **config["model"])
    elif name == "unet":
        model = UNet2DModule(denormalize=denormalize, **config["model"])
    else:
        raise ValueError(f"Model {name} not recognized.")

    # trainer
    config["trainer"]["max_epochs"] = 1
    trainer = L.Trainer(
        **config["trainer"],
        default_root_dir=output_dir,
    )

    tuner = Tuner(trainer)
    # Auto-scale batch size by growing it exponentially (default)
    tuner.scale_batch_size(model, datamodule=dm, mode="power")

    # training
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

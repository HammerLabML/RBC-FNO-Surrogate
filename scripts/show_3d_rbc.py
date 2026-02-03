import hydra
import torch
from rbc_fno_surrogate.data import RBCDatamodule3D
from rbc_fno_surrogate.utils.vis_3d import animation_3d


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_rbc")
def main(config):
    dm = RBCDatamodule3D(**config.data)
    dm.setup(stage="test")
    denorm = dm.datasets["test"].denormalize_batch

    idx = 0
    for _, ground_truth in dm.test_dataloader():
        state = denorm(ground_truth).cpu()
        noise = torch.randn_like(state) * 0.01
        fake = state + noise

        animation_3d(
            gt=state[0].numpy(),
            pred=fake[0].numpy(),
            feature="T",
            anim_dir=config.paths.output_dir + "/animations",
            anim_name=f"test_{idx}.mp4",
        )
        idx += 1
        break


if __name__ == "__main__":
    main()

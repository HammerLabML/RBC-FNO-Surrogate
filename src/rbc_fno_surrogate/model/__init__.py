from .fno2d import FNO2DModule
from .fno3d import FNO3DModule
from .lran2d import LRAN2DModule
from .lran3d import LRAN3DModule
from .ae2d import Autoencoder2DModule
from .ae3d import Autoencoder3DModule
from .unet2d import UNet2DModule
from .unet3d import UNet3DModule

__all__ = [
    "FNO2DModule",
    "PINO2DModule",
    "FNO3DModule",
    "cFNO2DModule",
    "LRAN2DModule",
    "LRAN3DModule",
    "Autoencoder2DModule",
    "Autoencoder3DModule",
    "UNet2DModule",
    "UNet3DModule",
]

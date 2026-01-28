from typing import List, Tuple
from collections import OrderedDict

import torch.nn as nn
from torch import Tensor


class Autoencoder2D(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        channels: List[int],
        kernel_size: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.input_channel = input_channel
        self.channels = channels
        self.kernel_size = kernel_size

        # Build models
        self.encoder = _Encoder(input_channel, channels, kernel_size, activation)
        self.decoder = _Decoder(input_channel, channels, kernel_size, activation)

        # linear layers
        self.encoder_linear = nn.Linear(self.encoder.out_dim, latent_dimension)
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dimension, self.encoder.out_dim),
            activation(),
        )

    def load_weights(self, ckpt: dict, freeze: bool) -> None:
        # Check hyperparameters
        params = ckpt["hyper_parameters"]
        assert self.input_channel == params["input_channel"], (
            f"'input_channel' does not match. ({self.input_channel} != {params['input_channel']})"
        )
        assert self.channels == params["channels"], (
            f"'channels' does not match. ({self.channels} != {params['channels']})"
        )
        assert self.kernel_size == params["kernel_size"], (
            f"'kernel_size' does not match. ({self.kernel_size} != {params['kernel_size']})"
        )

        # Load weights
        state = ckpt["state_dict"]
        encoder_weights = {
            k.replace("autoencoder.encoder.", ""): v
            for k, v in state.items()
            if k.startswith("autoencoder.encoder.")
        }
        decoder_weights = {
            k.replace("autoencoder.decoder.", ""): v
            for k, v in state.items()
            if k.startswith("autoencoder.decoder.")
        }
        self.encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)

        # Freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_linear(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.decoder_linear(z))


class _Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: List[int],
        kernel_size: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        # Build net
        layers = OrderedDict()
        for index, out_channels in enumerate(channels):
            layers[f"conv_{index}"] = nn.Conv2d(
                in_channels=input_channel if index == 0 else channels[index - 1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            layers[f"activation_{index}"] = activation()
            layers[f"pool_{index}"] = nn.MaxPool2d(kernel_size=(2, 2))
        layers["flatten"] = nn.Flatten()
        self.net = nn.Sequential(layers)

        # output dimension
        self.out_dim = (96 * 64 * channels[-1]) // (4 ** len(channels))

    def forward(self, x) -> Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: List[int],
        kernel_size: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        # Build net
        layers = OrderedDict()
        for index, out_channels in enumerate(reversed(channels)):
            layers[f"deconv_{index}"] = nn.ConvTranspose2d(
                in_channels=channels[-index - 1] if index == 0 else channels[-index],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=2,
                output_padding=1,
            )
            layers[f"activation_{index}"] = activation()

        layers["final_deconv"] = nn.ConvTranspose2d(
            in_channels=channels[0],
            out_channels=input_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.net = nn.Sequential(layers)

        # determine latent size
        self.xs = 96 // (2 ** len(channels))
        self.ys = 64 // (2 ** len(channels))

    def forward(self, x) -> Tensor:
        x = x.reshape(x.shape[0], -1, self.ys, self.xs)
        return self.net(x)

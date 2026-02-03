import copy
import torch
import torch.nn.functional as F
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, *ks):
        super().__init__()
        if len(ks) == 1:
            self.mp = nn.MaxPool1d(ks, ks)
        elif len(ks) == 2:
            self.mp = nn.MaxPool2d(ks, ks)
        elif len(ks) == 3:
            self.mp = nn.MaxPool3d(ks, ks)
        else:
            raise Exception()

    def forward(self, x):
        return self.mp(x)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding="same",
        padding_mode="zeros",
        dims=2,
        nl=nn.ReLU(),
        include_norm=True,
    ):
        super().__init__()
        conv_cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        # self.norm = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dims-1](num_features = out_channels, momentum = 0.001)

        self.norm = None
        if include_norm:
            norm_cls = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][
                dims - 1
            ]
            self.norm = norm_cls(num_features=out_channels, eps=1.0)

        self.nl = nl
        self.conv1 = conv_cls(
            in_channels, out_channels, 3, padding=padding, padding_mode=padding_mode
        )
        self.conv2 = conv_cls(
            out_channels, out_channels, 3, padding=padding, padding_mode=padding_mode
        )

    def forward(self, x):
        x = self.conv1(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.nl(x)
        x = self.conv2(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding,
        dims=2,
        nl=nn.ReLU(),
        include_norm=True,
    ):
        super().__init__()
        cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dc = DoubleConv(
            in_channels,
            out_channels,
            padding="valid",
            dims=dims,
            nl=nl,
            include_norm=include_norm,
        )
        self.conv = cls(in_channels, out_channels, kernel_size=1)
        self.padding = padding
        self.dims = dims
        self._pads = []
        for pm, dim in zip(self.padding, range(self.dims)):
            pad = [0] * (2 * self.dims)
            pad_idx = 2 * (self.dims - 1 - dim)
            pad[pad_idx] = 2
            pad[pad_idx + 1] = 2
            mode = "constant" if pm == "zeros" else pm
            self._pads.append((pad, mode))

    def forward(self, x):
        x1 = x
        for pad, mode in self._pads:
            if mode == "constant":
                x1 = F.pad(x1, pad, mode=mode, value=0)
            else:
                x1 = F.pad(x1, pad, mode=mode)

        y = self.conv(x)
        z = self.dc(x1)

        return y + z


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128),
        padding=("zeros", "zeros"),
        nl=nn.ReLU(),
    ):
        super().__init__()
        self.padding = padding
        self.dims = len(self.padding)
        assert self.dims in [1, 2, 3]

        features = [in_channels] + list(features) + [out_channels]
        self.pool = MaxPool(*[2] * self.dims)
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(
                    features[k],
                    features[k + 1],
                    padding=self.padding,
                    nl=nl,
                    dims=self.dims,
                )
                for k in range(len(features) - 1)
            ]
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            x = self.pool(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(32, 64),
        padding=("zeros", "zeros"),
        nl=nn.ReLU(),
    ):
        super().__init__()
        self.padding = padding
        self.dims = len(self.padding)
        assert self.dims in [1, 2, 3]

        self.encoders = torch.nn.ModuleList([])
        self.bridges = torch.nn.ModuleList([])
        self.decoders = torch.nn.ModuleList([])
        self.up = nn.Upsample(scale_factor=tuple([2] * self.dims), mode="nearest")
        self.pool = MaxPool(*[2] * self.dims)
        self.nl = nl

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_top_unet_block(in_channels, features[0], out_channels)
        for k in range(len(features) - 1):
            self.append_unet_block(features[k + 1])

    def forward(self, x):
        x_bridged = []
        for k in range(len(self.encoders)):
            x = self.encoders[k](x)
            x_bridged.append(self.bridges[k](x))
            if k < len(self.encoders) - 1:
                x = self.pool(x)
        for k in range(len(self.decoders) - 1, -1, -1):
            if k == len(self.decoders) - 1:
                x_up = torch.zeros_like(x_bridged[k])
            x = self.decoders[k](x_bridged[k] + x_up)
            if k > 0:
                x_up = self.up(x)
        return x

    def add_top_unet_block(self, in_channels, bridge_channels, out_channels):
        encoder = ResNetBlock(
            in_channels,
            bridge_channels,
            padding=self.padding,
            nl=self.nl,
            dims=self.dims,
            include_norm=False,
        )
        bridge = ResNetBlock(
            bridge_channels,
            bridge_channels,
            padding=self.padding,
            nl=self.nl,
            dims=self.dims,
            include_norm=False,
        )
        decoder = ResNetBlock(
            bridge_channels,
            out_channels,
            padding=self.padding,
            nl=self.nl,
            dims=self.dims,
            include_norm=False,
        )

        self.encoders.append(encoder)
        self.bridges.append(bridge)
        self.decoders.append(decoder)

    def append_unet_block(self, bridge_channels):
        top_channels = self.encoders[-1].out_channels

        encoder = ResNetBlock(
            top_channels,
            bridge_channels,
            padding=self.padding,
            nl=self.nl,
            dims=self.dims,
            include_norm=False,
        )
        bridge = ResNetBlock(
            bridge_channels,
            bridge_channels,
            padding=self.padding,
            nl=self.nl,
            dims=self.dims,
            include_norm=False,
        )
        decoder = ResNetBlock(
            bridge_channels,
            top_channels,
            padding=self.padding,
            nl=self.nl,
            dims=self.dims,
            include_norm=False,
        )

        self.encoders.append(encoder)
        self.bridges.append(bridge)
        self.decoders.append(decoder)

    def clone_and_adapt(self, additional_in_channels):
        cpy = copy.deepcopy(self)

        l1 = self.encoders[0].dc.conv1
        l1_ = type(l1)(
            l1.in_channels + additional_in_channels,
            l1.out_channels,
            kernel_size=l1.kernel_size,
            padding=l1.padding,
            padding_mode=l1.padding_mode,
        )
        l1_.weight.data[:, :-additional_in_channels] = l1.weight.data.clone()
        l1_.bias.data = l1.bias.data.clone()
        cpy.encoders[0].dc.conv1 = l1_

        l1 = self.encoders[0].conv
        l1_ = type(l1)(
            l1.in_channels + additional_in_channels,
            l1.out_channels,
            kernel_size=l1.kernel_size,
            padding=l1.padding,
            padding_mode=l1.padding_mode,
        )
        l1_.weight.data[:, :-additional_in_channels] = l1.weight.data.clone()
        l1_.bias.data = l1.bias.data.clone()
        cpy.encoders[0].conv = l1_

        return cpy

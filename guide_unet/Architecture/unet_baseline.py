import torch.nn as nn

from .unet_parts import DoubleConv, Down, OutConv, Up


class UNetBaseline(nn.Module):
    VALID_SKIP_MASK_LAYERS = ("up1", "up2", "up3", "up4")

    def __init__(
        self,
        n_channels=4,
        n_classes=4,
        bilinear=False,
        skip_mask_mode="normal",
        skip_mask_layers=None,
        skip_scale_gamma=1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip_mask_mode = "normal"
        self.skip_mask_layers = tuple()
        self.skip_scale_gamma = 1.0

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.set_skip_mask_mode(
            skip_mask_mode,
            skip_mask_layers=skip_mask_layers,
            skip_scale_gamma=skip_scale_gamma,
        )

    def _normalize_skip_mask_layers(self, layers):
        if layers is None:
            return tuple(self.VALID_SKIP_MASK_LAYERS)
        if isinstance(layers, str):
            text = layers.strip()
            if text == "" or text.lower() == "none":
                return tuple()
            if text.lower() == "all":
                return tuple(self.VALID_SKIP_MASK_LAYERS)
            layers = [item.strip() for item in text.split(",") if item.strip()]
        normalized = tuple(str(item) for item in layers)
        invalid = sorted(set(normalized) - set(self.VALID_SKIP_MASK_LAYERS))
        if invalid:
            raise ValueError("Invalid skip_mask_layers: {}".format(", ".join(invalid)))
        return normalized

    def set_skip_mask_mode(self, mode, skip_mask_layers=None, skip_scale_gamma=1.0):
        mode = str(mode)
        if mode not in ("normal", "zero", "scale"):
            raise ValueError("skip_mask_mode must be 'normal', 'zero', or 'scale', got {}".format(mode))
        self.skip_mask_mode = mode
        self.skip_mask_layers = tuple() if mode == "normal" else self._normalize_skip_mask_layers(skip_mask_layers)
        self.skip_scale_gamma = float(skip_scale_gamma)

    def set_skip_mask_layers(self, layers):
        self.skip_mask_layers = self._normalize_skip_mask_layers(layers)

    def _skip_input(self, layer_name, skip_feat):
        if self.skip_mask_mode == "zero" and str(layer_name) in self.skip_mask_layers:
            return skip_feat.new_zeros(skip_feat.shape)
        if self.skip_mask_mode == "scale" and str(layer_name) in self.skip_mask_layers:
            return skip_feat * float(self.skip_scale_gamma)
        return skip_feat

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        encoder = [x2, x3, x4, x5]

        d1 = self.up1(x5, self._skip_input("up1", x4))
        bridge_x = d1
        d2 = self.up2(d1, self._skip_input("up2", x3))
        d3 = self.up3(d2, self._skip_input("up3", x2))
        d4 = self.up4(d3, self._skip_input("up4", x1))

        decoder = [d1, d2, d3]
        logits = self.outc(d4)
        return logits, bridge_x, encoder, decoder


def get_model(
    n_channels=4,
    n_classes=4,
    bilinear=False,
    skip_mask_mode="normal",
    skip_mask_layers=None,
    skip_scale_gamma=1.0,
):
    return UNetBaseline(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        skip_mask_mode=skip_mask_mode,
        skip_mask_layers=skip_mask_layers,
        skip_scale_gamma=skip_scale_gamma,
    )

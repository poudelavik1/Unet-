'Models for hairline crack segmentation.'

from __future__ import annotations
from .common import DEFAULT_ENCODER, F, copy, math, nn, torch

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__(ConvBNReLU(input_channels, output_channels, stride=stride),
                         ConvBNReLU(output_channels, output_channels))


class ThinLineResponse(nn.Module):
    """Fixed morphological black-hat and white top-hat on grayscale input.

    Black-hat (closing - image) responds to dark structures narrower than the
    kernel; top-hat (image - opening) to bright ones. Both are needed: in EH104
    many cracks appear lighter than the concrete (deposits in the crack), not
    darker. The module has no parameters, so it is a thin-line prior at no data cost.
    """

    def __init__(self, kernel_sizes=(7, 15)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.output_channels = 2 * len(self.kernel_sizes)
        self.register_buffer("luma", torch.tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1))

    def forward(self, rgb):
        gray = (rgb * self.luma).sum(1, keepdim=True)
        responses = []
        for size in self.kernel_sizes:
            closed = -F.max_pool2d(-F.max_pool2d(gray, size, 1, size // 2), size, 1, size // 2)
            opened = F.max_pool2d(-F.max_pool2d(-gray, size, 1, size // 2), size, 1, size // 2)
            responses.extend((closed - gray, gray - opened))
        return torch.cat(responses, 1)


def smp_library():
    """segmentation_models_pytorch, imported when a network is built."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError("The network requires segmentation_models_pytorch: "
                           "python -m pip install segmentation-models-pytorch==0.5.0 timm") from exc
    return smp


def encoder_statistics(encoder_name: str) -> tuple[tuple, tuple]:
    """Mean and standard deviation of RGB in [0, 1] that the pretrained encoder expects.

    Most encoders were trained with the ImageNet statistics, but not all: the ``tf_`` ports of
    EfficientNet in timm use 0.5 for both. timm encoders (``tu-<name>``) carry the values in
    their pretrained configuration and smp's own encoders in smp's settings table.
    """
    smp = smp_library()
    if encoder_name.startswith("tu-"):
        import timm

        configuration = timm.models.get_pretrained_cfg(encoder_name[3:])
        if configuration is not None:
            return tuple(configuration.mean), tuple(configuration.std)
        return IMAGENET_MEAN, IMAGENET_STD
    settings = getattr(smp.encoders, "pretrained_settings", {}).get(encoder_name, {}).get("imagenet", {})
    return tuple(settings.get("mean", IMAGENET_MEAN)), tuple(settings.get("std", IMAGENET_STD))


class UNetPlusPlus(nn.Module):
    """UNet++ (Zhou et al., 2018) of segmentation_models_pytorch (smp) on a ConvNeXt or EfficientNet encoder.

    The encoder, the nested decoder with its dense skip connections and the 3x3 output
    convolution are smp's own (``get_encoder``, ``UnetPlusPlusDecoder``, ``SegmentationHead``),
    built and initialised as ``smp.UnetPlusPlus`` builds them. `encoder_name` is any smp
    encoder: ``tu-convnext_tiny`` (a timm model), ``efficientnet-b4`` (smp's own), ...

    The decoder nodes are named as in smp, ``x_<start>_<level>``: `level` 0 to 3 is the scale
    1/16 to 1/2 and `start` the depth the path comes from (0 = the 1/32 bottleneck). The 1/2-scale
    row ``x_3_3, x_2_3, x_1_3, x_0_3`` is X(0,1) to X(0,4) of the paper, and ``x_0_4`` the
    full-resolution node the output is taken from.

    Added to smp's network:

    * A 1/2-scale stem for encoders without 1/2-scale features. ConvNeXt downsamples four times
      in its first layer, and smp marks the missing scale by a feature with zero channels, on
      which the forward pass of ``smp.UnetPlusPlus`` fails. Two 3x3 convolutions on the image (the first
      with stride 2, `half_channels` wide, trained from scratch) supply the feature instead.
      EfficientNet has 1/2-scale features of its own and gets no such stem.
    * The hairline stem of the other trainers (`stem_channels` > 0): a full-resolution stem, fed
      with fixed black-hat and top-hat responses (`line_prior`), is the skip connection of the
      last decoder node. In smp that node upsamples from 1/2 scale without any skip, which is
      coarse for cracks one or two pixels wide. ``stem_channels=0`` leaves it as in smp.
    * Deep supervision as in the UNet++ paper (`deep_supervision`): a 1x1 output on each of the
      intermediate nodes X(0,1), X(0,2) and X(0,3).

    With an encoder that has 1/2-scale features, ``stem_channels=0`` and no deep supervision the
    network is ``smp.UnetPlusPlus`` itself, weight for weight.

    Input: RGB in [0, 1], height and width multiples of 32. Eval mode returns the crack logits
    (N, 1, H, W). Train mode returns ``(logits, auxiliary)``: the full-size logits of the
    deep-supervision outputs in the order X(0,1), X(0,2), X(0,3) (empty without deep supervision).
    """

    def __init__(
        self, encoder_name: str = DEFAULT_ENCODER, pretrained: bool = True,
        decoder_channels=(256, 128, 64, 32, 16), decoder_attention: str | None = "scse",
        decoder_interpolation: str = "nearest", half_channels: int = 48, stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, drop_path_rate: float = 0.0,
        prior_probability: float = 0.03,
    ):
        super().__init__()
        smp = smp_library()
        from segmentation_models_pytorch.base import SegmentationHead, initialization
        from segmentation_models_pytorch.decoders.unetplusplus.decoder import DecoderBlock, UnetPlusPlusDecoder

        decoder_channels = tuple(int(channels) for channels in decoder_channels)
        if len(decoder_channels) != 5:
            raise ValueError(f"decoder_channels needs five values (1/16 to 1/1 scale), not {decoder_channels}.")
        # Stochastic depth is an argument of timm models; smp's own encoders have theirs built in.
        timm_arguments = {"drop_path_rate": drop_path_rate} if encoder_name.startswith("tu-") and drop_path_rate > 0 else {}
        try:
            self.encoder = smp.encoders.get_encoder(
                encoder_name, in_channels=3, depth=5, weights="imagenet" if pretrained else None, **timm_arguments)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not build the {encoder_name} encoder with its pretrained weights ({exc}). Compute "
                    f"nodes cannot download: cache them on the login node with "
                    f"'ENCODER={encoder_name} bash scripts/setup_conda.sh', or pass --no-pretrained."
                ) from exc
            raise
        channels = list(self.encoder.out_channels)  # image, then the features at 1/2, 1/4, 1/8, 1/16, 1/32 scale
        if len(channels) != 6 or min(channels[2:]) < 1:
            raise ValueError(f"{encoder_name} does not give features at 1/4 to 1/32 scale (channels {channels}).")
        self.half_resolution = None
        if channels[1] == 0:  # ConvNeXt: no 1/2-scale features
            self.half_resolution = DoubleConv(3, half_channels, stride=2)
            channels[1] = half_channels
        # The nodes have the channels of decoder_channels or of the encoder feature of their scale (1/2 to 1/8).
        if decoder_attention == "scse" and min(*channels[1:4], *decoder_channels) < 16:
            raise ValueError("scSE attention divides the channels by 16: every decoder node needs at least 16 "
                             f"(encoder {channels[1:4]}, decoder {decoder_channels}). Use decoder_attention=None.")
        block_options = dict(use_norm="batchnorm", attention_type=decoder_attention,
                             interpolation_mode=decoder_interpolation)
        self.decoder = UnetPlusPlusDecoder(encoder_channels=channels, decoder_channels=decoder_channels,
                                           n_blocks=5, **block_options)
        self.depth = self.decoder.depth  # 4: the levels of the nested part, 1/16 to 1/2 scale
        self.thin_lines, self.full_resolution = None, None
        if stem_channels > 0:
            self.thin_lines = ThinLineResponse() if line_prior else None
            stem_inputs = 3 + (self.thin_lines.output_channels if line_prior else 0)
            self.full_resolution = DoubleConv(stem_inputs, stem_channels)
            # The last node, rebuilt with the stem as its skip connection.
            self.decoder.blocks[f"x_0_{self.depth}"] = DecoderBlock(
                self.decoder.in_channels[-1], stem_channels, decoder_channels[-1], **block_options)
        initialization.initialize_decoder(self.decoder)
        self.head = SegmentationHead(decoder_channels[-1], 1, kernel_size=3)
        initialization.initialize_head(self.head)
        # X(0,1) to X(0,3) have the channels of the 1/2-scale skip.
        self.auxiliary_heads = nn.ModuleList([
            nn.Conv2d(self.decoder.skip_channels[self.depth - 1], 1, 1) for _ in range(self.depth - 1)
        ]) if deep_supervision else None
        # Start near the crack prior so early training is not dominated by background.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head[0], *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        mean, std = encoder_statistics(encoder_name)
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1))

    def nodes(self, x) -> dict:
        """Every node of the nested decoder, by its smp name (``UnetPlusPlusDecoder.forward`` keeps only the last)."""
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(f"Height and width must be multiples of 32, not {tuple(x.shape[-2:])}.")
        normalized = (x - self.image_mean) / self.image_std
        features = list(self.encoder(normalized))
        if self.half_resolution is not None:
            features[1] = self.half_resolution(normalized)
        full = None
        if self.full_resolution is not None:
            stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
            full = self.full_resolution(stem_input)
        features = features[:0:-1]  # 1/32 first, without the image
        blocks, depth, nodes = self.decoder.blocks, self.depth, {}
        for layer in range(depth):
            for start in range(depth - layer):
                level = start + layer
                if layer == 0:
                    nodes[f"x_{start}_{level}"] = blocks[f"x_{start}_{level}"](features[start], features[start + 1])
                else:
                    skips = [nodes[f"x_{index}_{level}"] for index in range(start + 1, level + 1)]
                    nodes[f"x_{start}_{level}"] = blocks[f"x_{start}_{level}"](
                        nodes[f"x_{start}_{level - 1}"], torch.cat([*skips, features[level + 1]], 1))
        nodes[f"x_0_{depth}"] = blocks[f"x_0_{depth}"](nodes[f"x_0_{depth - 1}"], full)
        return nodes

    def forward(self, x):
        nodes = self.nodes(x)
        logits = self.head(nodes[f"x_0_{self.depth}"])
        if not self.training:
            return logits
        top = self.depth - 1  # the 1/2-scale row
        auxiliary = [] if self.auxiliary_heads is None else [
            F.interpolate(head(nodes[f"x_{top - index}_{top}"]), size=x.shape[-2:], mode="bilinear", align_corners=False)
            for index, head in enumerate(self.auxiliary_heads)
        ]
        return logits, auxiliary


# Checkpoint "architecture" names and the networks they rebuild.
ARCHITECTURES = {"unetplusplus": UNetPlusPlus}
# The arguments that decide the layers of the network: two runs share weights only if these agree.
NETWORK_ARGUMENTS = ("encoder_name", "decoder_channels", "decoder_attention", "decoder_interpolation",
                     "half_channels", "stem_channels", "line_prior", "deep_supervision")


def different_arguments(saved: dict, wanted: dict) -> dict:
    """The NETWORK_ARGUMENTS in which a checkpoint's network differs from the one wanted, with the saved values."""
    return {key: saved[key] for key in NETWORK_ARGUMENTS
            if key in saved and key in wanted and json_value(saved[key]) != json_value(wanted[key])}


def json_value(value):
    """Lists and tuples compare equal: a checkpoint read back holds what was saved as either."""
    return list(value) if isinstance(value, (list, tuple)) else value


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild the network of a best.pt for inference scripts.

    The architecture kwargs in the checkpoint carry the encoder and the decoder options. The
    model is called as ``model(images)`` with RGB in [0, 1] and returns the crack logits.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = ARCHITECTURES[architecture["name"]](pretrained=False, **architecture["kwargs"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


class ModelEMA:
    """Exponential moving average of weights (and BN statistics)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


__all__ = [
    'IMAGENET_MEAN', 'IMAGENET_STD', 'ConvBNReLU', 'DoubleConv', 'ThinLineResponse', 'smp_library',
    'encoder_statistics', 'UNetPlusPlus', 'ARCHITECTURES', 'NETWORK_ARGUMENTS', 'different_arguments',
    'json_value', 'load_model_from_checkpoint', 'ModelEMA',
]

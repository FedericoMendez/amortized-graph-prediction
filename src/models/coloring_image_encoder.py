"""Selectable ResNet18 and SegFormer encoders for Coloring RGB images."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class ResNetBasicBlock(nn.Module):
    """The two-convolution residual block used by ResNet18."""

    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.normalization_1 = nn.BatchNorm2d(output_channels)
        self.convolution_2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization_2 = nn.BatchNorm2d(output_channels)
        self.projection = (
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
            if stride != 1 or input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.projection(inputs)
        hidden = functional.relu(self.normalization_1(self.convolution_1(inputs)))
        hidden = self.normalization_2(self.convolution_2(hidden))
        return functional.relu(hidden + residual)


class TruncatedResNet18(nn.Module):
    """ResNet18 through layer2, with the first max-pool removed."""

    output_channels = 128

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer_1 = nn.Sequential(
            ResNetBasicBlock(64, 64),
            ResNetBasicBlock(64, 64),
        )
        self.layer_2 = nn.Sequential(
            ResNetBasicBlock(64, 128, stride=2),
            ResNetBasicBlock(128, 128),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layer_2(self.layer_1(self.stem(images)))


def _two_dimensional_positions(
    height: int,
    width: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build deterministic sine/cosine row and column position features."""

    if dimension % 4:
        raise ValueError("2D positional encoding requires a dimension divisible by 4.")
    coordinate_dimension = dimension // 2
    frequency = torch.exp(
        torch.arange(0, coordinate_dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / coordinate_dimension)
    )

    def encode(length: int) -> Tensor:
        angles = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        angles = angles * frequency.unsqueeze(0)
        result = torch.empty(
            (length, coordinate_dimension), device=device, dtype=torch.float32
        )
        result[:, 0::2] = angles.sin()
        result[:, 1::2] = angles.cos()
        return result

    rows = encode(height)[:, None, :].expand(height, width, -1)
    columns = encode(width)[None, :, :].expand(height, width, -1)
    return (
        torch.cat((rows, columns), dim=-1)
        .reshape(height * width, dimension)
        .to(dtype)
    )


class ResNetColoringImageEncoder(nn.Module):
    """Convert an RGB image to spatial Transformer tokens.

    The convolutional trunk follows the Coloring encoder described by
    Any2Graph: ResNet18 without its first max-pool or last two residual stages.
    Large feature maps are adaptively pooled before quadratic self-attention.
    """

    def __init__(
        self,
        *,
        d_token_input: int,
        n_heads: int,
        n_layers: int,
        d_token_input_feedforward: int,
        dropout: float,
        max_feature_grid_size: int,
    ) -> None:
        super().__init__()
        if d_token_input % 4:
            raise ValueError("Coloring d_token_input must be divisible by 4.")
        if max_feature_grid_size < 1:
            raise ValueError("max_feature_grid_size must be positive.")
        self.max_feature_grid_size = max_feature_grid_size
        self.convolutional_encoder = TruncatedResNet18()
        self.input_projection = nn.Sequential(
            nn.Linear(TruncatedResNet18.output_channels, d_token_input),
            nn.GELU(),
            nn.LayerNorm(d_token_input),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_token_input,
            nhead=n_heads,
            dim_feedforward=d_token_input_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_token_input),
            enable_nested_tensor=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Coloring images must have shape [batch, 3, height, width]."
            )
        features = self.convolutional_encoder(images)
        height = min(features.shape[-2], self.max_feature_grid_size)
        width = min(features.shape[-1], self.max_feature_grid_size)
        if features.shape[-2:] != (height, width):
            features = functional.adaptive_avg_pool2d(features, (height, width))
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.input_projection(tokens)
        positions = _two_dimensional_positions(
            height,
            width,
            tokens.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return self.encoder(tokens + positions.unsqueeze(0))


class DropPath(nn.Module):
    """Per-sample stochastic depth used by the SegFormer MiT blocks."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep_probability)
        return inputs * mask / keep_probability


class OverlappingPatchEmbedding(nn.Module):
    """Convolutional overlapping patch projection from the SegFormer encoder."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.normalization = nn.LayerNorm(output_channels)

    def forward(self, inputs: Tensor) -> tuple[Tensor, int, int]:
        features = self.projection(inputs)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return self.normalization(tokens), height, width


class SpatialReductionAttention(nn.Module):
    """Multi-head attention with convolutionally reduced keys and values."""

    def __init__(
        self,
        dimension: int,
        n_heads: int,
        spatial_reduction_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension % n_heads:
            raise ValueError("SegFormer stage width must be divisible by its heads.")
        self.n_heads = n_heads
        self.head_dimension = dimension // n_heads
        self.scale = self.head_dimension**-0.5
        self.query = nn.Linear(dimension, dimension)
        self.key_value = nn.Linear(dimension, 2 * dimension)
        self.spatial_reduction = (
            nn.Conv2d(
                dimension,
                dimension,
                kernel_size=spatial_reduction_ratio,
                stride=spatial_reduction_ratio,
            )
            if spatial_reduction_ratio > 1
            else None
        )
        self.reduced_normalization = (
            nn.LayerNorm(dimension)
            if spatial_reduction_ratio > 1
            else nn.Identity()
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dimension, dimension)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch, token_count, dimension = tokens.shape
        query = self.query(tokens).reshape(
            batch, token_count, self.n_heads, self.head_dimension
        )
        query = query.transpose(1, 2)
        key_value_tokens = tokens
        if self.spatial_reduction is not None:
            features = tokens.transpose(1, 2).reshape(
                batch, dimension, height, width
            )
            features = self.spatial_reduction(features)
            key_value_tokens = features.flatten(2).transpose(1, 2)
            key_value_tokens = self.reduced_normalization(key_value_tokens)
        key_value = self.key_value(key_value_tokens).reshape(
            batch,
            key_value_tokens.shape[1],
            2,
            self.n_heads,
            self.head_dimension,
        )
        key_value = key_value.permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch, token_count, dimension)
        return self.output_dropout(self.output(output))


class MixFeedForward(nn.Module):
    """SegFormer Mix-FFN with a depthwise spatial convolution."""

    def __init__(self, dimension: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden_dimension = dimension * expansion
        self.input = nn.Linear(dimension, hidden_dimension)
        self.depthwise = nn.Conv2d(
            hidden_dimension,
            hidden_dimension,
            kernel_size=3,
            padding=1,
            groups=hidden_dimension,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dimension, dimension)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch = tokens.shape[0]
        hidden = self.input(tokens)
        hidden = hidden.transpose(1, 2).reshape(batch, -1, height, width)
        hidden = self.depthwise(hidden).flatten(2).transpose(1, 2)
        hidden = self.dropout(self.activation(hidden))
        return self.dropout(self.output(hidden))


class MixTransformerBlock(nn.Module):
    """Pre-normalized attention and Mix-FFN residual block."""

    def __init__(
        self,
        dimension: int,
        n_heads: int,
        spatial_reduction_ratio: int,
        *,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.attention_normalization = nn.LayerNorm(dimension)
        self.attention = SpatialReductionAttention(
            dimension,
            n_heads,
            spatial_reduction_ratio,
            dropout,
        )
        self.feedforward_normalization = nn.LayerNorm(dimension)
        self.feedforward = MixFeedForward(dimension, expansion=4, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        tokens = tokens + self.drop_path(
            self.attention(self.attention_normalization(tokens), height, width)
        )
        return tokens + self.drop_path(
            self.feedforward(
                self.feedforward_normalization(tokens), height, width
            )
        )


class SegFormerB1Backbone(nn.Module):
    """Native MiT-B1 backbone returning four hierarchical feature maps."""

    stage_dimensions = (64, 128, 320, 512)
    stage_heads = (1, 2, 5, 8)
    stage_depths = (2, 2, 2, 2)
    spatial_reduction_ratios = (8, 4, 2, 1)

    def __init__(self, *, dropout: float, maximum_drop_path: float = 0.1) -> None:
        super().__init__()
        input_dimensions = (3,) + self.stage_dimensions[:-1]
        self.patch_embeddings = nn.ModuleList(
            [
                OverlappingPatchEmbedding(
                    input_dimension,
                    output_dimension,
                    kernel_size=7 if index == 0 else 3,
                    stride=4 if index == 0 else 2,
                    padding=3 if index == 0 else 1,
                )
                for index, (input_dimension, output_dimension) in enumerate(
                    zip(input_dimensions, self.stage_dimensions, strict=True)
                )
            ]
        )
        drop_paths = torch.linspace(
            0.0, maximum_drop_path, sum(self.stage_depths)
        ).tolist()
        offset = 0
        stages = []
        for dimension, heads, depth, reduction in zip(
            self.stage_dimensions,
            self.stage_heads,
            self.stage_depths,
            self.spatial_reduction_ratios,
            strict=True,
        ):
            stages.append(
                nn.ModuleList(
                    [
                        MixTransformerBlock(
                            dimension,
                            heads,
                            reduction,
                            dropout=dropout,
                            drop_path=drop_paths[offset + index],
                        )
                        for index in range(depth)
                    ]
                )
            )
            offset += depth
        self.stages = nn.ModuleList(stages)
        self.output_normalizations = nn.ModuleList(
            [nn.LayerNorm(dimension) for dimension in self.stage_dimensions]
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1]
            fan_out *= module.out_channels / module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> list[Tensor]:
        if images.shape[-2] < 32 or images.shape[-1] < 32:
            raise ValueError(
                "SegFormer-B1 requires Coloring images of at least 32 x 32 pixels."
            )
        hidden = images
        outputs = []
        for patch_embedding, blocks, normalization in zip(
            self.patch_embeddings,
            self.stages,
            self.output_normalizations,
            strict=True,
        ):
            tokens, height, width = patch_embedding(hidden)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = normalization(tokens)
            hidden = tokens.transpose(1, 2).reshape(
                images.shape[0], -1, height, width
            )
            outputs.append(hidden)
        return outputs


class SegFormerColoringImageEncoder(nn.Module):
    """Randomly initialized MiT-B1 fusion encoder for image-to-graph prediction."""

    def __init__(
        self,
        *,
        d_token_input: int,
        dropout: float,
        max_feature_grid_size: int,
    ) -> None:
        super().__init__()
        if max_feature_grid_size < 1:
            raise ValueError("max_feature_grid_size must be positive.")
        self.max_feature_grid_size = max_feature_grid_size
        self.backbone = SegFormerB1Backbone(dropout=dropout)
        self.stage_projections = nn.ModuleList(
            [
                nn.Conv2d(dimension, d_token_input, kernel_size=1)
                for dimension in self.backbone.stage_dimensions
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                len(self.backbone.stage_dimensions) * d_token_input,
                d_token_input,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(d_token_input),
            nn.GELU(),
        )
        self.output_normalization = nn.LayerNorm(d_token_input)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Coloring images must have shape [batch, 3, height, width]."
            )
        features = self.backbone(images)
        height = min(features[0].shape[-2], self.max_feature_grid_size)
        width = min(features[0].shape[-1], self.max_feature_grid_size)
        fused_features = []
        for feature_map, projection in zip(
            features, self.stage_projections, strict=True
        ):
            projected = projection(feature_map)
            if projected.shape[-2:] != (height, width):
                projected = functional.interpolate(
                    projected,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            fused_features.append(projected)
        fused = self.fusion(torch.cat(fused_features, dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        return self.output_normalization(tokens)


def build_coloring_image_encoder(
    *,
    encoder_type: str,
    d_token_input: int,
    n_heads: int,
    n_layers: int,
    d_token_input_feedforward: int,
    dropout: float,
    max_feature_grid_size: int,
) -> nn.Module:
    """Construct the selected Coloring input encoder."""

    if encoder_type == "resnet18":
        return ResNetColoringImageEncoder(
            d_token_input=d_token_input,
            n_heads=n_heads,
            n_layers=n_layers,
            d_token_input_feedforward=d_token_input_feedforward,
            dropout=dropout,
            max_feature_grid_size=max_feature_grid_size,
        )
    if encoder_type == "segformer_b1":
        return SegFormerColoringImageEncoder(
            d_token_input=d_token_input,
            dropout=dropout,
            max_feature_grid_size=max_feature_grid_size,
        )
    raise ValueError(f"Unsupported Coloring image encoder: {encoder_type!r}.")


# Backward-compatible public name for the original Coloring encoder.
ColoringImageEncoder = ResNetColoringImageEncoder

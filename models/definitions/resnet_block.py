import torch
import torch.nn as nn

class PositionalNorm(nn.LayerNorm):
    """PositionalNorm is a normalization layer used for 3D image inputs that
    normalizes exclusively across the channels dimension.
    https://arxiv.org/abs/1907.04312
    """

    def forward(self, x):
        # The input is of shape (B, C, H, W). Transpose the input so that the
        # channels are pushed to the last dimension and then run the standard
        # LayerNorm layer.
        x = x.permute(0, 2, 3, 1).contiguous()
        out = super().forward(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class ResBlock(nn.Module):
    """Residual block following the "bottleneck" architecture as described in
    https://arxiv.org/abs/1512.03385. See Figure 5.
    The residual blocks are defined following the "pre-activation" technique
    as described in https://arxiv.org/abs/1603.05027.

    The residual block applies a number of convolutions represented as F(x) and
    then skip connects the input to produce the output F(x)+x. The residual
    block can also be used to upscale or downscale the input by doubling or
    halving the spatial dimensions, respectively. Scaling is performed by the
    "bottleneck" layer of the block. If the residual block changes the number of
    channels, or the spatial dimensions are up- or down-scaled, then the input
    is also transformed into the desired shape for the addition operation.
    """

    def __init__(self, in_chan, out_chan, scale="same"):
        """Init a Residual block.

        Args:
            in_chan: int
                Number of channels of the input tensor.
            out_chan: int
                Number of channels of the output tensor.
            scale: string, optional
                One of ["same", "upscale", "downscale"].
                Upscale or downscale by half the spatial dimensions of the
                input tensor. Default is "same", i.e., no scaling.
        """
        super().__init__()
        assert scale in ["same", "upscale", "downscale"]
        if scale == "same":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, padding="same")
            stride = 1
        elif scale == "downscale":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, stride=2, padding=1)
            stride = 2
        elif scale == "upscale":
            bottleneck = nn.ConvTranspose2d(in_chan//2, in_chan//2, kernel_size=4, stride=2, padding=1)
            stride = 1

        # The residual block employs the bottleneck architecture as described
        # in Sec 4. under the paragraph "Deeper Bottleneck Architectures" of the
        # original paper introducing the ResNet architecture.
        # The block uses a stack of three layers: `1x1`, `3x3` (`4x4`), `1x1`
        # convolutions. The first `1x1` reduces (in half) the number of channels
        # before the expensive `3x3` (`4x4`) convolution. The second `1x1`
        # up-scales the channels to the requested output channel size.
            
        self.block = nn.Sequential(
            # 1x1 convolution
            PositionalNorm(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, in_chan//2, kernel_size=1),

            # 3x3 convolution if same or downscale, 4x4 transposed convolution if upscale
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            bottleneck,

            # 1x1 convolution
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            nn.Conv2d(in_chan//2, out_chan, kernel_size=1),
        )

        # If channels or spatial dimensions are modified then transform the
        # input into the desired shape, otherwise use a simple identity layer.
        self.id = nn.Identity()
        
        if in_chan != out_chan or scale == "downscale":
            # We will downscale by applying a strided `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride),
            )
        if scale == "upscale":
            # We will upscale by applying a nearest-neighbor upsample.
            # Channels are again modified using a `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x):
        return self.block(x) + self.id(x)

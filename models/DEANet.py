import torch
from torch import nn
from einops import rearrange

# Channel Attention Module (CGA)
class ChannelAttention_CGA(nn.Module):
    # Reference: SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        # Depthwise Convolutions with different kernel sizes
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # Base convolution
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)  # Vertical convolution
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)  # Horizontal convolution

        # Larger kernel convolutions for multi-scale feature extraction
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)  # 1x1 convolution for feature integration

    def forward(self, x):
        u = x.clone()  # Clone the input for residual connection
        attn = self.conv0(x)  # Initial convolution

        # Multi-scale attention extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        # Combine all attention maps
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)  # Process with 1x1 convolution
        return attn * u  # Weighted multiplication with the input


# Spatial Attention Module (CGA)
class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        # 7x7 convolution with 'reflect' padding for spatial attention
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        # Compute global features through average pooling and max pooling
        x_avg = torch.mean(x, dim=1, keepdim=True)  # Average pooling
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling
        x2 = torch.concat([x_avg, x_max], dim=1)  # Concatenate along channel dimension
        sattn = self.sa(x2)  # Extract spatial attention using convolution
        return sattn


# Pixel Attention Module (CGA)
class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        # Pixel-level attention using depthwise convolution
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for attention weighting

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # Add temporal dimension (B, C, 1, H, W)
        pattn1 = pattn1.unsqueeze(dim=2)  # Add temporal dimension
        x2 = torch.cat([x, pattn1], dim=2)  # Concatenate along temporal dimension (B, C, 2, H, W)
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')  # Rearrange dimensions for convolution input
        pattn2 = self.pa2(x2)  # Extract pixel attention
        pattn2 = self.sigmoid(pattn2)  # Apply Sigmoid activation
        return pattn2


# Combined CGA Fusion Module
class CGAFusion(nn.Module):
    def __init__(self, dim):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()  # Spatial Attention
        self.ca = ChannelAttention_CGA(dim)  # Channel Attention
        self.pa = PixelAttention_CGA(dim)  # Pixel Attention
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)  # 1x1 convolution for final feature integration
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, data):
        x, y = data  # Two input feature maps
        initial = x + y  # Initial feature fusion
        cattn = self.ca(initial)  # Channel attention
        sattn = self.sa(initial)  # Spatial attention
        pattn1 = sattn + cattn  # Combine spatial and channel attention
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  # Pixel attention with Sigmoid
        # Final weighted feature fusion
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)  # Final 1x1 convolution
        return result

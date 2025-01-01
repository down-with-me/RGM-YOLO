import torch
import torch.nn as nn
import numpy as np

# Space-to-Depth Operation: Rearranges spatial blocks into the depth dimension
class space_to_depth(nn.Module):
    def forward(self, x):
        # Rearrange spatial data into depth by selecting pixels in a 2x2 grid
        # Combine these blocks into the channel dimension
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


# Large Kernel Spatial-Kernel Attention Block (LSKblock)
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution with a 5x5 kernel
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # Depthwise convolution with a 7x7 kernel and dilation for larger receptive field
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
        # Reduces the number of channels (dim -> dim // 2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        
        # Squeeze operation for combining attention information
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        
        # Final convolution to restore channel dimension
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # Extract features using the two convolutions
        attn1 = self.conv0(x)  # Standard depthwise convolution
        attn2 = self.conv_spatial(attn1)  # Large spatial kernel convolution
        
        # Reduce channels to half
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        # Concatenate the two attention features
        attn = torch.cat([attn1, attn2], dim=1)
        
        # Compute average pooling and max pooling along the channel dimension
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        
        # Combine pooling results and pass through a squeeze convolution
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()  # Apply sigmoid activation
        
        # Weighted combination of attention maps
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        
        # Final convolution to integrate features
        attn = self.conv(attn)
        return x * attn  # Element-wise multiplication with the input


# LSKblock Attention Module
class LSKblockAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        # Rearranges spatial blocks into the depth dimension
        self.space_to_depth = space_to_depth()
        
        # 1x1 convolution to project features
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        
        # Non-linear activation
        self.activation = nn.GELU()
        
        # Spatial gating unit using LSKblock
        self.spatial_gating_unit = LSKblock(d_model)
        
        # Another 1x1 convolution for final projection
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # Clone input for residual connection
        shorcut = x.clone()
        
        # Apply space-to-depth transformation
        x = self.space_to_depth(x)
        
        # Project features to the same dimension
        x = self.proj_1(x)
        
        # Apply non-linear activation
        x = self.activation(x)
        
        # Pass through the spatial gating unit (attention mechanism)
        x = self.spatial_gating_unit(x)
        
        # Final projection back to the original dimension
        x = self.proj_2(x)
        
        # Add residual connection
        x = x + shorcut
        return x

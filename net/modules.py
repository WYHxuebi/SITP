import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
import torch
import torch.nn.functional as F


class Mlp(nn.Module):
    """
    Swin Transformer Block - MLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    # view: [B, H / window_size, window_size, W / window_size, window_size, C]
    # permute: [B, H / window_size, W / window_size, window_size, window_size, C]
    # view: [B * (H / window_size) * (W / window_size), window_size, window_size, C] 
    #       = [B * num_windows, window_size, window_size, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    # windows: [B * num_windows, window_size, window_size, C] 
    #          = [B * (H / window_size) * (W / window_size), window_size, window_size, C]
    # view: [B, H / window_size, W / window_size, window_size, window_size, C]
    # permute: [B, H / window_size, window_size, W / window_size, window_size, C]
    # view: [B, H, W, C]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size                  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  

        # get pair-wise relative position index for each token inside the window
        # coords_flatten[:, :, None] - coords_flatten[:, None, :]: [2, window_size * window_size, window_size * window_size]
        # permute: [window_size * window_size, window_size * window_size, 2]
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                                   # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()             # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [num_windows * B, windows_size * windows_size, C]
        B_, N, C = x.shape
        
        device = x.get_device()
        if mask != None:
            mask = mask.to(device)

        # self.qkv: [num_windows * B, windows_size * windows_size, 3 * C]
        # reshape: [num_windows * B, windows_size * windows_size, 3, num_heads,  per_head_C]
        # permute: [3, num_windows * B, num_heads, windows_size * windows_size, per_head_C]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: [num_windows * B, num_heads, per_head_C, windows_size * windows_size]
        # attn: [num_windows * B, num_heads, windows_size * windows_size, windows_size * windows_size]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        # relative_position_index: [window_size * window_size, window_size * window_size]
        # view(-1): (window_size * window_size) * (window_size * window_size)
        # relative_position_bias_table: [(window_size * window_size) * (window_size * window_size), num_head]
        # view(): [window_size * window_size, window_size * window_size, num_head]
        # permute: [num_head, window_size * window_size, window_size * window_size]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # attn: [num_windows * B, num_heads, windows_size * windows_size, windows_size * windows_size]
        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(
                0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        # Adding a mask before Softmax will make the Softmax value approximately 0 for discontinuous regions.
        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            
            # Mask: [num_window, window_size * window_size, window_size * window_size]
            nW = mask.shape[0]

            # view: [B, num_windows, num_heads, windows_size * windows_size, windows_size * windows_size]
            # mask.unsqueeze(1): [1, num_windows, 1, windows_size * windows_size, windows_size * windows_size]
            # view: [num_windows * B, num_heads, windows_size * windows_size, windows_size * windows_size]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: [num_windows * B, num_heads, windows_size * windows_size, per_head_C]
        # transpose: [num_windows * B, windows_size * windows_size, num_heads, per_head_C]
        # reshape: [num_windows * B, windows_size * windows_size, C]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim

        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N

        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += N * self.dim * self.dim

        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # begin:end:stride
        # [B, H/2, W/2, C]
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]                # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]                # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]                # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]                # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)     # [B, H/2, W/2, 4*C]
        x = x.view(B, H*W//4, 4 * C)            # [B, H/2 * W/2, 4*C]

        # [B, H/2 * W/2, out_dim]
        x = self.norm(x)
        x = self.reduction(x)

        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x


    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super(PatchMerging4x, self).__init__()
        H, W = input_resolution
        self.patch_merging1 = PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_merging2 = PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H//2, W//2)
        return x


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super(PatchReverseMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        # [B, H * W, 4 * out_dim]
        # view: [B, H, W, 4 * out_dim]
        # permute: [B, 4 * out_dim, H, W]
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        # [B, 4 * out_dim, H, W]
        # [B, out_dim, 2 * H, 2 * W]
        # flatten: [B, out_dim, 2 * H * 2 * W]
        # permute: [B, 2 * H * 2 * W, out_dim]
        x = nn.PixelShuffle(2)(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchReverseMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super(PatchReverseMerging4x, self).__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)


    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H*2, W*2)
        return x


    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchEmbed(nn.Module):
    """
    Patch_Embedding层
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)   

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None


    def forward(self, x):
        [B, 3, 32, 32]
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x: [B, C, H, W]/[B, 3, 32, 32]
        # self.proj: [B, ED, H/2, W/2]/[B, 128, 16, 16]
        # flatten: [B, ED, H/2 * W/2]/[B, 128, 256]
        # transpose: [B, H/2 * W/2, ED]/[B, 256, 128]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
from net.modules import *
import torch
import torch.nn as nn


class SwinTransformerBlock(nn.Module):
    """
    SwinTransformerBlock
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:  
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        # Multi-head Attention within Window
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        # Layer Norm
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer)

        # Img_Mask
        if self.shift_size > 0:
            H, W = self.input_resolution

            # [1, H, W, 1]
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            # Creat Mask
            # window_partition: [num_Windows, window_size, window_size, 1]
            # view: [num_Windows, window_size * window_size]
            # unsqueeze(1): [num_Windows, 1, window_size * window_size]
            # unsqueeze(2):: [num_Windows, window_size * window_size, 1]
            # unsqueeze(2) - unsqueeze(1): [num_Windows, window_size * window_size, window_size * window_size]
            mask_windows = window_partition(img_mask, self.window_size)  
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x):
        """
        Forward Propagation
        """

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Layer Norm
        shortcut = x
        x = self.norm1(x)

        # Dimensional Transformation
        x = x.view(B, H, W, C)

        # cyclic shift
        # shift_size>0: SW-MSA
        # shift_size=0: 对应W-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        B_, N, C = x_windows.shape

        # W-MSA / SW-MSA
        # [num_Windows * B, window_size * window_size, C]
        attn_windows = self.attn(x_windows, add_token = False, mask = self.attn_mask)
        
        # merge windows
        # view: [num_Windows * B, window_size, window_size, C]
        # window_reverse: [B, H', W', C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


    def flops(self):
        flops = 0
        H, W = self.input_resolution

        # norm1
        flops += self.dim * H * W

        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio

        # norm2
        flops += self.dim * H * W
        return flops


    def update_mask(self, device):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda(device)
        else:
            pass


class BasicLayer(nn.Module):
    """
    basiclayer: Patch merging and the swin transformer are combined into a single basic layer.
    dim: Number of input channels. When i_layer equals 0, the number of input channels is 3.
    out_dim: Number of output channels.
    input_resolution: Input feature dimension.
    depth: Number of times the swin transformer is stacked.
    """
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim = out_dim,
                                 input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads = num_heads, 
                                 window_size = window_size,
                                 shift_size = 0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio = mlp_ratio,
                                 qkv_bias = qkv_bias, 
                                 qk_scale = qk_scale,
                                 norm_layer = norm_layer)
            for i in range(depth)])

        # patch merging
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim = dim, out_dim = out_dim, norm_layer = norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        """
        Forward Propagation
        """
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x


    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


    def update_resolution(self, H, W, device):
        """
        Update the input feature size for each stage
        """
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask(device)
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)


class AdaptiveModulator(nn.Module):
    """
    Channel ModNet
    SM Module
    """
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )


    def forward(self, snr):
        return self.fc(snr)


class WITT_Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True, bottleneck_dim=16):
        super(WITT_Encoder, self).__init__()

        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, 
                                      in_chans=self.in_chans, embed_dim=embed_dims[0],
                                      norm_layer=norm_layer if self.patch_norm else None)

        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims)-1], self.hidden_dim))

        # Channel ModNet
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims)-1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim = int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim = int(embed_dims[i_layer]),
                               input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth = depths[i_layer],
                               num_heads = num_heads[i_layer],
                               window_size = window_size,
                               mlp_ratio = self.mlp_ratio,
                               qkv_bias = qkv_bias, qk_scale = qk_scale,
                               norm_layer = norm_layer,
                               downsample = PatchMerging if i_layer != 0 else None)
            self.layers.append(layer)

        self.norm = norm_layer(embed_dims[-1])
        self.head_list = nn.Linear(embed_dims[-1], C)
        self.apply(self._init_weights)


    def forward(self, x, snr):
        """
        Forward Propagation
        """
        
        # [B, C, H, W]/[B, 3, 32, 32]
        B, C, H, W = x.size()
        device = x.get_device()

        # [B, H' * W', C']/[B, 256, 128]
        x = self.patch_embed(x)

        # [B, 16*16, 128] -> [B, 16*16, 128] -> [B, 8*8, 256]
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)

        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            # bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
            bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, x.size(1), -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val

        # [B, 8*8, 256] -> [B, 8*8, 96]
        x = self.head_list(x)

        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops


    def update_resolution(self, H, W, device):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)), device)


def create_encoder(**kwargs):
    model = WITT_Encoder(**kwargs)
    return model


if __name__ == '__main__':

    # # AFHQ
    # encoder_kwargs = dict(
    #     img_size=(192, 192), patch_size=2, in_chans=3,
    #     embed_dims=[128, 192, 256, 320], depths=[2, 2, 4, 2], num_heads=[4, 4, 6, 8], C=96,
    #     window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #     norm_layer=nn.LayerNorm, patch_norm=True)
    # input_image = torch.ones([2, 3, 192, 192]).cuda()

    encoder_kwargs = dict(
        img_size=(256, 256), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=96,
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True)
    input_image = torch.ones([2, 3, 256, 256]).cuda()

    model = create_encoder(**encoder_kwargs).cuda()
    y = model(x=input_image, snr=10)
    CBR = y.numel() / input_image.numel()
    print(y.size())
    print(CBR)
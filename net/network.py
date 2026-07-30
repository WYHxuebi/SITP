from loss.distortion import Distortion
from random import choice
import torch.nn as nn
import numpy as np
from net.decoder import *
from net.encoder import *
from net.lpips import LPIPS
from net.channel import Channel
from net.frame import FrameInterleaver
from net.digital import DigitalLink
from net.m2rmodule import M2RModule_Res


class SITP(nn.Module):
    """
    Main Network
    """
    def __init__(self, args, config):
        super(SITP, self).__init__()

        # Configurations
        self.args = args
        self.config = config

        # Obtain Encoder and Decoder Parameters
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs

        # Creating Encoders and Decoders
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)

        # Pseudo Masked Decoder
        self.Unet_pmd = M2RModule_Res(channel=args.C)

        # Image Quality Assessment
        self.distortion_loss = Distortion(args)
        self.LPIPS = LPIPS().eval()

        # Channel
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel

        # MSE
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.loss_fn = torch.nn.L1Loss()
        self.H = self.W = 0

        self.multiple_snr = config.multiple_snr
        self.frame_loss_rates = config.frame_loss_rates
        frame_loss_rate = choice(self.frame_loss_rates)
            
        # Frame Processer
        self.FrameInterleaver = FrameInterleaver(args.frame_len, args.interleave_mode, 
                                                 config.channel,  frame_loss_rate)
        
        # Other Parameters
        self.downsample = config.downsample
        self.local_rank = args.local_rank
        self.num_bits = 4
        self.levels = 2 ** self.num_bits
        self.action_scale = self.levels / 2.
        self.action_bias = 0.5       
        self.bit_flip_ratio = 0.0125
        self.mask_ratio = 2 * self.bit_flip_ratio

        # Digital Communication
        self.DigitalLink = DigitalLink()


    def tensor_to_binary_tensor(self, tensor):

        # Ensure the input tensor has the desired data type (integer)
        tensor = tensor.to(dtype=torch.int) + (2**(self.num_bits-1) - 1)

        # Calculate the maximum value a single element can have based on L
        max_value = 2**self.num_bits - 1
        if torch.any(tensor < 0):
            tensor[torch.where(tensor < 0)]=0

        # Check if any element in the tensor is out of range
        if torch.any((tensor < 0) | (tensor > max_value)):
            raise ValueError("Input tensor contains elements out of range.")
        
        # Create a binary mask for shifting and extracting each bit
        bit_mask = torch.tensor([2**i for i in range(self.num_bits-1, -1, -1)], dtype=torch.int, device=tensor.device)

        # Create a binary tensor by bitwise AND operation with the bit mask
        binary_tensor = (tensor.unsqueeze(-1) & bit_mask).gt(0).to(dtype=torch.int)

        return binary_tensor


    def binary_tensor_to_tensor(self, binary_tensor):

        # Ensure the input tensor has the desired data type (integer)
        binary_tensor = binary_tensor.to(dtype=torch.int)

        # Calculate the number of bits (L)
        L = self.num_bits

        # Convert binary strings to integers for each batch element
        tensor = torch.sum(binary_tensor * 2**(torch.arange(L-1, -1, step=-1, dtype=torch.int, device=binary_tensor.device)), dim=-1)

        return tensor - 2**(self.num_bits - 1) + 1


    def bit_flip(self, x):
        random_array = torch.rand(x.shape)
        flipped_indices = random_array < self.bit_flip_ratio
        x[flipped_indices] = 1 - x[flipped_indices]
        return x


    def qam16_modulate(self, bits):
        return self.DigitalLink.qam16_modulate(bits)
    

    def qam16_demodulate(self, symbols):
        return self.DigitalLink.qam16_demodulate(symbols)


    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr, self.args.train_stage)
        return noisy_feature
    

    def mask_generate(self, x):
        cache = torch.rand(x.shape)
        mask = cache > self.mask_ratio
        return mask.to(x.device)
    

    def add_nosiychannel(self, feature, given_SNR=None):
        chan_param = given_SNR
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        return noisy_feature


    def encoding(self, input_image, given_SNR):

        B, _, H, W = input_image.shape

        # Update the input feature size for each stage
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W, self.local_rank)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample), self.local_rank)
            self.H = H
            self.W = W

        # Image passed through encoder
        feature = self.encoder(input_image, given_SNR)
        feature_scale = torch.tanh(feature) * self.action_scale + self.action_bias

        return feature_scale
        

    def decoding_native(self, noisy_feature, input_image, given_SNR):

        # Calculate the compression ratio
        CBR = noisy_feature.numel() / 2 / input_image.numel()

        # Features Decoded
        recon_image = self.decoder(noisy_feature, given_SNR)

        # Calculate MSE loss
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)

        # Computational image quality assessment
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        lpips = torch.mean(self.LPIPS(input_image.contiguous(), recon_image.contiguous()))
        
        return recon_image, CBR, given_SNR, mse.mean(), lpips.item(), loss_G.mean()
    

    def decoding_mask(self, noisy_feature, Tx, input_image, given_SNR, mask_flag=True):

        # Calculate the compression ratio
        CBR = noisy_feature.numel() / 2 / input_image.numel()

        # Masking
        if mask_flag:
            mask = self.mask_generate(noisy_feature)
            noisy_feature = mask * noisy_feature

        # Dimensional transformation
        noisy_feature = noisy_feature.transpose(1, 2)
        B, C, HW = noisy_feature.shape
        H = W = np.sqrt(HW).astype(int)
        noisy_feature = noisy_feature.view(B, C, H, W)

        # Noise reduction
        Rx = self.Unet_pmd(noisy_feature)
        Rx = Rx.view(B, C, H * W)
        Rx = Rx.transpose(1, 2)

        # Calculate denoising loss
        loss_pmd = self.loss_fn(Rx, Tx)

        # Features Decoded
        recon_image = self.decoder(Rx, given_SNR)

        # Calculate MSE loss
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)

        # Calculate the loss function
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        loss_G = loss_G + loss_pmd

        # Calculate the LPIPS metric
        lpips = torch.mean(self.LPIPS(input_image.contiguous(), recon_image.contiguous()))
        
        return recon_image, CBR, given_SNR, mse.mean(), lpips.item(), loss_G.mean()
    

    def frame_interleave(self, feature_bits):
        frames, padded, frame_pad_len = self.FrameInterleaver.interleave(feature_bits)
        return frames, padded, frame_pad_len
    

    def frame_loss(self, noisy_frame):
        framesloss, mask = self.FrameInterleaver.lossframe(noisy_frame)
        return framesloss, mask


    def frame_deinterleave(self, framesloss, padded, frame_pad_len, feature_bits_shape):
        noisy_feature_bits = self.FrameInterleaver.deinterleave(framesloss, padded, frame_pad_len, feature_bits_shape)
        return noisy_feature_bits
    

    def forward(self, input_image, given_SNR = None):
        B, _, H, W = input_image.shape

        # Update the input feature size for each stage
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W, self.local_rank)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample), self.local_rank)
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        # Image passed through encoder
        feature = self.encoder(input_image, chan_param)

        # Quantitative Operation
        feature_q = self.quantize(feature)

        # Calculate the compression ratio
        CBR = feature_q.numel() / 2 / input_image.numel()

        # Source coding
        feature_bits = self.DigitalLink.quant2bits(feature_q)
        feature_bits_shape = feature_bits.shape
        feature_bits_string = feature_bits.view(-1)

        # Data packets
        frames, padded, frame_pad_len = self.FrameInterleaver.interleave(feature_bits_string)

        # Channel coding
        feature_ldpc_bits = self.DigitalLink.encode_ldpc(frames)

        # Digital modulation
        feature_symbols, modulate_pad_len = self.DigitalLink.qam16_modulate(feature_ldpc_bits)

        # Features passing through the channel
        noisy_feature_symbols = self.add_nosiychannel(feature_symbols, chan_param)

        # Digital demodulation
        noisy_feature_ldpc_bits = self.DigitalLink.qam16_demodulate(noisy_feature_symbols, modulate_pad_len)

        # Channel decoding
        noisy_frame = self.DigitalLink.decode_ldpc(noisy_feature_ldpc_bits)

        # Packet loss
        if self.args.train_stage == 1:
            framesloss = noisy_frame
        elif self.args.train_stage == 2:
            framesloss, mask = self.FrameInterleaver.lossframe(noisy_frame)
        else:
            raise ValueError(f"Unknown train stage: {self.args.train_stage}")

        # Deframe
        noisy_feature_bits_string = self.FrameInterleaver.deinterleave(framesloss, padded, frame_pad_len, feature_ldpc_bits.shape)

        # Source decoding
        noisy_feature_bits = noisy_feature_bits_string.view(feature_bits_shape)
        noisy_feature = self.DigitalLink.bits2quant(noisy_feature_bits)
        noisy_feature = feature + (noisy_feature - feature).detach()

        # Dequantization
        feature_noise_hat = self.dequantize(noisy_feature)

        # Features Decoded
        recon_image = self.decoder(feature_noise_hat, chan_param)

        # Calculate MSE loss
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)

        # Computational image quality assessment
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        lpips = torch.mean(self.LPIPS(input_image.contiguous(), recon_image.contiguous()))

        return recon_image, CBR, chan_param, mse.mean(), lpips, loss_G.mean()
        

if __name__ == "__main__":

    class args():
        training = True
        train_stage = 1
        trainset = 'AFHQ'
        distortion_metric = 'MSE'
        channel_type = 'awgn'
        frame_loss_type = 'iid'
        C = 96
        multiple_snr = '0, 5, 10, 15'
        batch_size = 32
        frame_len = 256
        interleave_mode = 'random'
        frame_loss_rate = '0.1, 0.2, 0.3'
        local_rank = 0


    class config():

        seed = 0
        pass_channel = False
        CUDA = True
        device = torch.device("cuda", args.local_rank)
        norm = False
        lpips = True
        isTrain = True
        channel = args.frame_loss_type

        # AFHQ
        image_dims = (3, 256, 256)
        batch_size = args.batch_size
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True)
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm)   
        
    args = args()
    config = config()
    net = SITP(args, config).to(config.device)

    x = torch.randn(args.batch_size, 3, 256, 256).to(config.device)
    y, CBR, SNR, MSE, LPIPS, Loss_G = net(x, given_SNR=10)
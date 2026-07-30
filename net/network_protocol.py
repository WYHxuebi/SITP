from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.lpips import LPIPS
from net.channel import Channel
from random import choice
import torch.nn as nn
import numpy as np
from net.frame import FrameInterleaver
from net.digital import DigitalLink
from net.m2rmodule import M2RModule_Res


class SITP_Protocol(nn.Module):
    """
    Main Network
    """
    def __init__(self, args, config):
        super(SITP_Protocol, self).__init__()

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

        self.multiple_snr = config.SNR_db
        frame_loss_rate = 0.0
            
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

        if self.config.protocol == 'STCP':
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature
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

        # Frame
        if self.args.train_stage == 1:
            frames = feature_q
        elif self.args.train_stage == 2:
            frames, padded, pad_len = self.FrameInterleaver.interleave(feature_q)
        else:
            raise ValueError(f"Unknown train stage: {self.args.train_stage}")

        # Features passing through the channel
        if self.config.protocol == 'STCP':
            if self.pass_channel:
                noisy_frame = self.feature_pass_channel(frames, chan_param)
            else:
                noisy_frame = frames
        else:
            noisy_frame = frames

        # Packet loss
        if self.args.train_stage == 1:
            framesloss = noisy_frame
        elif self.args.train_stage == 2:
            framesloss, mask = self.FrameInterleaver.lossframe(noisy_frame)
        else:
            raise ValueError(f"Unknown train stage: {self.args.train_stage}")

        # Deframe
        if self.args.train_stage == 1:
            noisy_feature = framesloss
        elif self.args.train_stage == 2:
            noisy_feature = self.FrameInterleaver.deinterleave(framesloss, padded, pad_len, feature.shape)
        else:
            raise ValueError(f"Unknown train stage: {self.args.train_stage}")
        
        # Dequantization
        feature_noise_hat = self.dequantize(noisy_feature)

        # Features Decoded
        recon_image = self.decoder(feature_noise_hat, chan_param)

        # Calculate MSE loss
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)

        # Computational image quality assessment
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
    
        if self.lpips_using:
            lpips = torch.mean(self.LPIPS(input_image.contiguous(), recon_image.contiguous()))
            return recon_image, CBR, chan_param, mse.mean(), lpips, loss_G.mean()
        else:
            return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()
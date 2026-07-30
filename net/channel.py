import torch.nn as nn
import numpy as np
import torch


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """
    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device


    def gaussian_noise_layer(self, input_layer, std):
        """
        Gaussian White Noise
        """
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer)).to(self.device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer)).to(self.device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise


    def rayleigh_noise_layer(self, input_layer, std):
        """
        Rayleigh Noise
        """
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer)).to(self.device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer)).to(self.device)
        noise = noise_real + 1j * noise_imag

        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
        h = h.to(self.device)

        return input_layer * h + noise


    def complex_normalize(self, x, power):
        """
        Power Normalization
        """
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr


    def forward(self, input, chan_param, avg_pwr=False, train_stage=1):
        """
        Forward Propagation
        """
        
        # Signal Power Normalization
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        # Pass Through Channel
        channel_output = self.complex_forward(channel_tx, chan_param)

        # Signal Power Recovery
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)


    def complex_forward(self, channel_in, chan_param):
        
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        # Gaussian channel
        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx, std=sigma)
            return chan_output

        # Rayleigh Channel
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx, std=sigma)
            return chan_output
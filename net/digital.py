import torch
import math
import os
from commpy.channelcoding.ldpc import get_ldpc_code_params


class DigitalLink():
    def __init__(self):

        self.gray2level_axis = torch.tensor([-3.0, -1.0, +3.0, +1.0])
        self.binidx2bits = torch.tensor([
            [0., 0.],  # bin 0 -> -3
            [0., 1.],  # bin 1 -> -1
            [1., 1.],  # bin 2 -> +1
            [1., 0.],  # bin 3 -> +3
        ])

        # Average energy normalization coefficient (16QAM average |s|^2 = 10)
        self.norm = math.sqrt(10.0)

        # LDPC Encoding Parameters
        self.ldpc_design_file = os.path.join(os.path.dirname(__file__), '1440.720.txt')
        ldpc_param = get_ldpc_code_params(self.ldpc_design_file)
    

    # Digital Modulation
    def qam16_modulate(self, bits):
        
        device = bits.device
        last = bits.shape[-1]

        assert last % 4 == 0, f"The last dimension must be divisible by 4, currently it is {last}"
        n_sym = last // 4
        bits4 = bits.reshape(*bits.shape[:-1], n_sym, 4).to(torch.float32)

        # I-axis index = b3b2, Q-axis index = b1b0 (0..3)
        i_idx = (bits4[..., 0].long() << 1) | bits4[..., 1].long()
        q_idx = (bits4[..., 2].long() << 1) | bits4[..., 3].long()

        # Mapping to levels (Gray): [-3, -1, +3, +1] (by idx 0..3)
        g2l = self.gray2level_axis.to(device)
        I = g2l[i_idx]
        Q = g2l[q_idx]

        # Form a complex symbol and normalize the unit average power.
        symbols = (I + 1j * Q).to(torch.complex64) / self.norm  # [B,HW,C]

        if n_sym == 1:
            symbols = symbols.squeeze(-1)

        return symbols

    # Digital Demodulation
    def qam16_demodulate(self, noisy_symbols):

        device = noisy_symbols.device

        # Denormalization to the unnormalized constellation domain facilitates decision-making using a fixed threshold.
        s = (noisy_symbols * self.norm).to(torch.complex64)

        # I/Q
        I = s.real  # [B,HW,C]
        Q = s.imag  # [B,HW,C]


        # To avoid discontinuity issues: construct the bin index (0..3) using comparisons.
        i_bin = (I >= -2).to(torch.int64) + (I >= 0).to(torch.int64) + (I >= 2).to(torch.int64)
        q_bin = (Q >= -2).to(torch.int64) + (Q >= 0).to(torch.int64) + (Q >= 2).to(torch.int64)

        # bin index -> Gray two bits
        bin2b = self.binidx2bits.to(device)
        i_bits = bin2b[i_bin]  # [B,HW,C,2] : (b3,b2)
        q_bits = bin2b[q_bin]  # [B,HW,C,2] : (b1,b0)

        # (b3 b2 b1 b0)
        bits = torch.cat([i_bits, q_bits], dim=-1).to(torch.float32)  # [B,HW,C,4]

        if noisy_symbols.dim() == 2:
            # [T, K] -> [T, 4K]
            T, K = noisy_symbols.shape
            return bits.reshape(T, K * 4)
        elif noisy_symbols.dim() == 3:
            # [B, HW, C] -> [B, HW, C, 4]
            return bits
    

    # Channel Coding
    def encode_ldpc(self, bits):
        # During training, it can directly return to itself; 
        # during inference, it can replace the real LDPC library.
        return bits
    

    # Channel Decoding
    def decode_ldpc(self, bits):
        return bits


if __name__ == "__main__":

    torch.manual_seed(42)

    B = 2           # batch size
    HW = 256
    C = 96
    N = 16
    num_bits = 4
    snr_db = 10

    # DigitalLink
    link = DigitalLink()

    # Construct Random Quantization Symbols (0~1)
    feature_q = torch.randint(0, 2, (B, HW, C, num_bits))

    print("Input quantization symbol:")
    print(feature_q.shape)

    modulate_16qam_signal = link.qam16_modulate(feature_q)
    print(modulate_16qam_signal.shape)

    demofulate_16qam_signal = link.qam16_demodulate(modulate_16qam_signal)
    print(demofulate_16qam_signal.shape)
    print(torch.sum(demofulate_16qam_signal != feature_q).item())
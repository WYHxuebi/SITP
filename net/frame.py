import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from math import lgamma, log, exp
import math


class FrameInterleaver():
    def __init__(self, frame_len=8, interleave_mode='sequential', channel='ge', loss_rate=0.3, generator=None):
        self.frame_len = frame_len
        self.interleave_mode = interleave_mode
        self.loss_rate = loss_rate
        self.shuffle_idx = None
        self.channel = channel
        self.init_state = 'G'
        self.gen = generator

        self.ge_params = {'p_g': 0.01, 'p_b': loss_rate, 'p_gb': 0.10, 'p_bg': 0.30}
        self.p_g = float(self.ge_params.get('p_g'))      # Good State Packet Loss Probability
        self.p_b = float(self.ge_params.get('p_b'))      # Bad State Packet Loss Probability
        self.p_gb = float(self.ge_params.get('p_gb'))    # Good to Bad Transition Probability
        self.p_bg = float(self.ge_params.get('p_bg'))    # Bad to Good Transition Probability
        self.burst_pkts = 24*22

    def _rand(self, shape=(), device='cpu'):
        if self.gen is None:
            return torch.rand(shape, device=device)
        else:
            return torch.rand(shape, device=device, generator=self.gen)
    
    def interleave(self, x):

        flat = x.view(-1)
        sequence_length = flat.size(0)

        if sequence_length % self.frame_len != 0:
            pad_len = self.frame_len - (sequence_length % self.frame_len)
            flat = F.pad(flat, (0, pad_len), value=0.0)
            padded = True
        else:
            pad_len = 0
            padded = False

        sequence_length = flat.size(0)
        T = sequence_length // self.frame_len
        
        if self.interleave_mode == 'sequential':
            ordered = flat
        elif self.interleave_mode == 'random':
            self.shuffle_idx = torch.randperm(sequence_length, device=x.device)
            ordered = flat[self.shuffle_idx]
        else:
            raise ValueError(f"Unknown mode: {self.interleave_mode}")
        
        interleaved = ordered.view(T, self.frame_len)

        return interleaved, padded, pad_len
    
    
    def _ge_states(self, T, device):
        states = torch.empty(T, dtype=torch.long, device=device)
        state = 0 if self.init_state.upper().startswith('G') else 1
        for t in range(T):
            states[t] = state
            u = self._rand((), device=device).item()
            if state == 0:
                # G -> B with prob p_gb
                state = 1 if u < self.p_gb else 0
            else:
                # B -> G with prob p_bg
                state = 0 if u < self.p_bg else 1
        return states

    def lossframe(self, frame):
        T, L = frame.shape
        device = frame.device
        self.p_b = self.loss_rate

        if self.channel == 'iid':
            mask = (self._rand((T,), device=device) >= self.loss_rate).float()

        elif self.channel == 'ge':
            states = self._ge_states(T, device)
            p_loss = torch.where(states == 0,
                                 torch.full((T,), self.p_g, device=device),
                                 torch.full((T,), self.p_b, device=device))
            mask = (self._rand((T,), device=device) >= p_loss).float()

        elif self.channel == 'burst':

            # Packet Loss Rate of Initialized Good and Bad Windows
            p_g = self.p_g 
            p_b = self.p_b

            # Default Bad Window Length (Num. of Packets)
            burst_pkts = getattr(self, "burst_pkts", 24*22)

            # Randomly Select the Location of the Bad Window
            start = torch.randint(low=0, high= T - burst_pkts + 1, size=(1,), device=device).item()
            end = start + burst_pkts

            # Constructing Packet Loss Probability
            p_loss = torch.full((T,), p_g, device=device)
            p_loss[start:end] = p_b

            mask = (self._rand((T,), device=device) >= p_loss).float()

        else:
            raise ValueError("channel must be 'iid' or 'ge' !")
        
        frame_out = frame * mask.view(T, 1).to(frame.dtype)

        return frame_out, mask

    
    def deinterleave(self, frames, padded, pad_len, original_shape=None):
        T, L = frames.shape
        flat = frames.view(-1)

        if self.interleave_mode == 'sequential':
            restored = flat
        elif self.interleave_mode == 'random':
            reverse_idx = torch.argsort(self.shuffle_idx)
            restored = flat[reverse_idx]
        else:
            raise ValueError(f"Unknown mode: {self.interleave_mode}")
        
        if padded and pad_len > 0:
            restored = restored[:-pad_len]

        if original_shape is not None:
            restored = restored.view(*original_shape)

        return restored


# Q Function
def Q_function(x):
    return 0.5 * torch.erfc(x / math.sqrt(2))


# QAM Theoretical Bit Error Rate Curve
def ber_mqam(M, snr_db):
    k = math.log2(M)
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(snr_db, dtype=torch.float32)
    snr = 10 ** (snr_db / 10)
    coeff = 4.0 / k * (1 - 1 / math.sqrt(M))
    q_arg = torch.sqrt((3 * k * snr) / (M - 1))
    Pb = coeff * Q_function(q_arg)
    return Pb.item()


# Calculate the Binomial Distribution
def binom_cdf_leq(n: int, k: int, p: float) -> float:

    if p <= 0.0: 
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    
    log1mp = log(1.0 - p)

    def logpmf(i):
        return (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log1mp)
    
    i_star = int(np.clip(int((n + 1) * p), 0, n))
    max_log = logpmf(i_star)

    s = 0.0
    for i in range(0, k+1):
        s += exp(logpmf(i) - max_log)
    output = np.clip(s * exp(max_log), 0.0, 1.0)
    
    return output


# Calculate QSTP Packet Loss Rate
def sitp_fl_rate_calu(M, snr, frame_len):

    N_phy_header = 64
    t_phy_sync = 3
    N_phy_sync = 11

    r_mac = 32
    N_mac_head = 112
    two_pow_r_crc = 2.0 ** r_mac

    K_ip = 2
    N_ip_head = 160

    N_sitp_head = 64
    two_pow_r_chs = 2.0 ** r_mac

    N_app_header = 24

    N_total_head = N_phy_header + K_ip * N_ip_head + N_app_header

    # Bit Error Rate
    BER = ber_mqam(M, snr)

    # Physical Layer Synchronization Part Pass Probability
    Pr_phy_pass = binom_cdf_leq(N_phy_sync, t_phy_sync, BER)

    # Transport Layer SITP Pass Probability
    Pr_sitp_pass = 1.0 - (1.0 - ((1.0 - BER) ** N_sitp_head) * (1.0 - 1.0 / two_pow_r_chs))

    # Data Link Layer CRC Check Pass Probability
    Pr_mac_pass = 1.0 - (1.0 - ((1 - BER) ** N_mac_head)) * (1.0 - 1.0 / two_pow_r_crc)

    # Head Pass Probability
    Pr_head_pass = (1.0 - BER) ** N_total_head

    # Calculate Packet Loss Rate
    Pr_fl = 1 - Pr_phy_pass * Pr_sitp_pass * Pr_mac_pass * Pr_head_pass

    return Pr_fl


# Calculate UDP Packet Loss Rate
def udp_fl_rate_calu(M, snr, frame_len):

    # Physical Layer
    N_phy_header = 64
    t_phy_sync = 3
    N_phy_sync = 11

    # Data Link Layer
    r_mac = 32
    N_mac_head = 112
    two_pow_r_crc = 2.0 ** r_mac

    # Network Layer
    K_ip = 2
    N_ip_head = 160

    # Transport Layer
    N_udp_head = 64
    N_udp_data = frame_len
    two_pow_r_chs = 2.0 ** r_mac

    N_app_header = 24

    # Head length
    N_total_head = N_phy_header + K_ip * N_ip_head + N_app_header

    # Bit Error Rate
    BER = ber_mqam(M, snr)

    # Physical Layer Synchronization Pass Probability
    Pr_phy_pass = binom_cdf_leq(N_phy_sync, t_phy_sync, BER)
    
    # Transport Layer Verification Pass Probability
    N_udp_cov = N_udp_head + N_udp_data
    Pr_udp_pass = 1.0 - (1.0 - ((1.0 - BER) ** N_udp_cov) * (1.0 - 1.0 / two_pow_r_chs))
    
    # Data Link Layer CRC Check Pass Probability
    Pr_mac_pass = 1.0 - (1.0 - (1 - BER) ** N_mac_head) * (1.0 - 1.0 / two_pow_r_crc)
    
    # Header Pass Probability
    Pr_head_pass = (1.0 - BER) ** N_total_head

    # Calculate Packet Loss Rate
    Pr_fl = 1 - Pr_phy_pass * Pr_udp_pass * Pr_mac_pass * Pr_head_pass
    
    return Pr_fl


def tcp_fl_rate_calu(M, snr, frame_len):

    # Physical Layer
    N_phy_header = 64
    t_phy_sync = 3
    N_phy_sync = 11

    # Data Link Layer
    r_mac = 32
    N_mac_head = 112
    two_pow_r_crc = 2.0 ** r_mac

    # Network Layer
    K_ip = 2
    N_ip_head = 160

    # Transport Layer
    N_tcp_head = 224
    N_tcp_data = frame_len
    tcp_retries = 5
    two_pow_r_chs = 2.0 ** r_mac

    N_app_header = 24

    # Head length
    N_total_head = N_phy_header + K_ip * N_ip_head + N_app_header

    # Bit Error Rate
    BER = ber_mqam(M, snr)

    # Physical Layer Synchronization Pass Probability
    Pr_phy_pass = binom_cdf_leq(N_phy_sync, t_phy_sync, BER)

    # Transport Layer Verification Pass Probability
    N_trans_cov = N_tcp_head + N_tcp_data
    Pr_trans_pass = 1.0 - (1.0 - ((1.0 - BER) ** N_trans_cov) * (1.0 - 1.0 / two_pow_r_chs))
    Pr_trans_ack_pass = 1.0 - (1.0 - ((1.0 - BER) ** N_tcp_head) * (1.0 - 1.0 / two_pow_r_chs))

    # MAC Verification
    Pr_mac_pass = 1.0 - (1.0 - (1 - BER) ** N_mac_head) * \
        (1.0 - 1.0 / two_pow_r_crc)

    # Header Pass Probability
    Pr_head_pass = (1.0 - BER) ** N_total_head

    # Single Transmission Success Rate
    P_tcp_data = Pr_phy_pass * Pr_trans_pass * Pr_mac_pass * Pr_head_pass
    P_tcp_ack = Pr_phy_pass * Pr_trans_ack_pass * Pr_mac_pass * Pr_head_pass

    # Single Transmission Failure Rate
    P_fl_one = 1 - P_tcp_ack * P_tcp_data

    # Considering the Maximum Number of Retransmissions
    P_fl_tcp = P_fl_one ** tcp_retries

    return P_fl_tcp


def fl_rate_calu(protocol, M, snr, frame_len):
    if protocol.upper() == 'SITP':
        return sitp_fl_rate_calu(M, snr, frame_len)
    elif protocol.upper() == 'UDP':
        return udp_fl_rate_calu(M, snr, frame_len)
    elif protocol.upper() == 'TCP':
        return tcp_fl_rate_calu(M, snr, frame_len)
    else:
        raise ValueError("protocol must be 'QSTP', 'UDP', or 'TCP'!")


if __name__ == '__main__':

    M = 16
    cita_qstp = [1, 2, 4, 1024]
    frame_len = 1024
    Nmax = 10
    SNR_db = np.arange(6, 15, 0.2)

    Pr_flrs_qstp = {cita: [] for cita in cita_qstp}
    Pr_flrs_udp = []
    Pr_flrs_tcp = []

    for snr_db in SNR_db:
        for cita in cita_qstp:
            Pr_flrs_qstp[cita].append(sitp_fl_rate_calu(M, snr_db, frame_len))
        Pr_fl_udp = udp_fl_rate_calu(M, snr_db, frame_len)
        Pr_fl_tcp = tcp_fl_rate_calu(M, snr_db, frame_len, Nmax)
        Pr_flrs_udp.append(Pr_fl_udp)
        Pr_flrs_tcp.append(Pr_fl_tcp)

    colors = ['b', 'orange', 'g', 'r', 'purple', 'brown']
    markers = ['o', '^', 's', 'D', '*', 'x']

    plt.figure(0)

    for idx, cita in enumerate(cita_qstp):
        plt.plot(SNR_db, Pr_flrs_qstp[cita], color=colors[idx], marker=markers[idx], label=f'QSTP-{cita_qstp[idx]}', linewidth=1.5, markersize=3)
    plt.plot(SNR_db, Pr_flrs_udp, color=colors[4], marker=markers[4], label='UDP', linewidth=1.5, markersize=3)
    plt.plot(SNR_db, Pr_flrs_tcp, color=colors[5], marker=markers[5], label='TCP', linewidth=1.5, markersize=3)


    tcp_vals = np.array(Pr_flrs_tcp)
    snr_vals = np.array(SNR_db)
    idx_left = np.argmax(tcp_vals < 0.99)
    snr_left = snr_vals[idx_left]
    idx_right = np.argmax(tcp_vals < 0.01)
    snr_right = snr_vals[idx_right]
    plt.axvline(x=snr_left, color='gray', linestyle='--', linewidth=1.0)
    plt.axvline(x=snr_right, color='gray', linestyle='--', linewidth=1.0)
    plt.text(snr_left, -0.065, f'{snr_left:.2f}', ha='center', va='top', color='gray')
    plt.text(snr_right, -0.065, f'{snr_right:.2f}', ha='center', va='top', color='gray')


    plt.xlabel("SNR (dB)", fontsize=15)
    plt.ylabel("Frame Loss Rate", fontsize=15)
    plt.title("FL vs SNR in QSTP/UDP/TCP", fontsize=20, fontweight='bold')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.savefig("./mertic/Compare_FL_SNR_SUT.png", dpi=1200)
    plt.show()
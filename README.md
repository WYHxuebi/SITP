<h1 align="center">SITP</h1>

<p align="center">
  <b>A High-Reliability Semantic Information Transport Protocol Without Retransmission for Semantic Communication</b>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11517511">
    <img src="https://img.shields.io/badge/Paper-IEEE%20TCOM-blue" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/YunhaoWang/SITP">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow" alt="Hugging Face Dataset">
  </a>
  <a href="https://www.modelscope.cn/models/wyh13114873863/SITP">
    <img src="https://img.shields.io/badge/ModelScope-Model-624AFF" alt="ModelScope Model">
  </a>
  <a href="https://github.com/WYHxuebi/SITP/stargazers">
    <img src="https://img.shields.io/github/stars/WYHxuebi/SITP?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/WYHxuebi/SITP/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/WYHxuebi/SITP" alt="License">
  </a>
  <a href="https://github.com/WYHxuebi/SITP/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/WYHxuebi/SITP" alt="Contributors">
  </a>
  <img src="https://img.shields.io/github/repo-size/WYHxuebi/SITP" alt="Repository size">
</p>

<p align="center">
  <em>SITP enables low-latency and high-reliability semantic communication by validating only packet headers, preserving corrupted payloads for reconstruction, and mitigating burst losses through cross-image feature interleaving.</em>
</p>

---

## 📰 1. News

- **[2026-07-31]** 🤗 The pretrained resources are available on [Hugging Face](https://huggingface.co/datasets/YunhaoWang/SITP) and [ModelScope](https://www.modelscope.cn/models/wyh13114873863/SITP).
- **[2026-07-30]** 🎉 We release the official implementation of **SITP: A High-Reliability Semantic Information Transport Protocol Without Retransmission for Semantic Communication**.
- **[2026-05-13]** 📄 SITP was accepted by **IEEE Transactions on Communications**.

> The repository provides the official implementation, pretrained models, training scripts, and evaluation code for SITP.

---

## 🖼️ 2. Overview Figure

<p align="center">
  <a href="./figures/Cross_Layer_Architecture.png">
    <img
      src="./figures/Cross_Layer_Architecture.png"
      alt="SITP Cross Layer Architecture"
      width="100%"
    >
  </a>
  <br>
  <em> Figure 1: The cross-layer architecture of digital semantic communication based on SITP. Note: AH, NH, DH, and PH denote the application-layer header, network header, data-link header, and physical-layer header, respectively.</em>
</p>

---

## 🔍 3. Overview

Conventional transport protocols such as TCP and UDP are designed around **bit-level reliability**. However, semantic communication can often reconstruct meaningful information even when the received semantic representation is partially corrupted.

To bridge this gap, we propose the **Semantic Information Transport Protocol (SITP)**, a transport-layer protocol specifically designed for semantic communication, which follows a simple principle:

> **Verify the header, preserve the payload.**

Unlike TCP, SITP removes connection establishment and retransmission mechanisms to reduce end-to-end latency. Unlike UDP, SITP does not discard an entire packet when the semantic payload contains bit errors. Instead, SITP verifies only the packet header and forwards the potentially corrupted payload to the semantic decoder.

The resulting system provides:

- **TCP-level reliability** without retransmission;
- **UDP-level latency** without connection setup;
- robust semantic reconstruction under packet corruption;
- improved resilience to burst packet losses through cross-image feature interleaving.

---


## 🚀 4. Key Contributions

### 4.1 Semantic Information Transport Protocol

SITP introduces a transport-layer design tailored to the error tolerance of semantic communication. By retaining payloads whose headers pass verification, SITP allows the semantic decoder to exploit residual information that would otherwise be discarded.

| Protocol | Handshake / Retrans. | Validation Coverage | Noisy Payload Retained |
|:---:|:---:|:---:|:---:|
| TCP | 3-way handshake and ACK retrans. | Header + payload | No |
| UDP | No handshake or retrans. | Header + payload | No |
| UDP-Lite | No handshake or retrans. | Header + partial payload | Partially |
| PR-SCTP | 4-way handshake and partial retrans. | Header + payload | Partially |
| **SITP (Ours)** | **No handshake or retrans.** | **Header only** | **Yes** |

### 4.2 Cross-layer Mathematical Model of Packet Loss
Based on SITP, we establish a unified cross-layer mathematical model to characterize the end-to-end packet-loss probability across the physical, data-link, network, transport, and application layers. By incorporating the relationship between the signal-to-noise ratio (SNR), bit-error rate $P_b$, and packet-loss probability, the model enables systematic analysis of semantic transmission performance over digital communication systems.

$$
P_{\mathrm{Cross\mbox{-}fail}}(P_b)
=1-(1-P_b)^{8(N_{\mathrm{PH}}+N_{\mathrm{NH}}+N_{\mathrm{AH}})}
\left[
\sum_{i=0}^{t_{\mathrm{sync}}}
{8N_{\mathrm{sync}} \choose i}
P_b^i
(1-P_b)^{8N_{\mathrm{sync}}-i}
\right]
\left(
1-
\left[
1-(1-P_b)^{8N_{\mathrm{DH}}}
\right]
(1-2^{-r_d})
\right)
\left(
1-
\left[
1-(1-P_b)^{8N_{\mathrm{SITP-HDR}}}
\right]
(1-2^{-r_s})
\right)
$$

### 4.3 Cross-Image Feature Interleaving

To mitigate consecutive burst losses, SITP incorporates a cross-image semantic feature-level interleaving mechanism. 

Instead of independently protecting each image, semantic features from multiple images are redistributed across packets, which prevents concentrated packet losses from causing complete semantic collapse in a single image.

---

## 🏗️ 5. Framework

<p align="center">
  <a href="./figures/The_overall_architecture.png">
    <img
      src="./figures/The_overall_architecture.png"
      alt="Overall architecture of the SITP-based digital semantic communication system"
      width="100%"
    >
  </a>
  <br>
  <em> Figure 2: The overall architecture of the proposed SITP-based digital semantic communication system for the burst-loss resilience.</em>
</p>

---

## 📦 6. Installation

Clone the repository:

```bash
git clone https://github.com/WYHxuebi/SITP.git
cd SITP
```

Install the required packages:

```bash
pip install -r requirements.txt
```

> The current implementation is designed for CUDA-enabled PyTorch environments.

---

## 📂 7. Dataset Preparation

The experiments in this repository use the following datasets:

- **AFHQ**
- **ImageNet-10**

The processed datasets can be downloaded from our Hugging Face dataset page:

- [Hugging Face Dataset: YunhaoWang/SITP](https://huggingface.co/datasets/YunhaoWang/SITP)

After downloading, please extract the files and organize them under the './dataset' directory as follows:
  <details>
    
    ```
    ./dataset
        /AFHQ:
            /train:
                /cat
                    /flickr_cat_000002.jpg 
                    /flickr_cat_000003.jpg
                    ...
                /dog
                    /flickr_dog_000002.jpg 
                    /flickr_dog_000003.jpg
                    ...
                /wild
                    /flickr_wild_000002.jpg 
                    /flickr_wild_000003.jpg
                    ...
            /val:
                /cat
                    /flickr_cat_000008.jpg 
                    /flickr_cat_000011.jpg
                    ...
                /dog
                    /flickr_dog_000043.jpg 
                    /flickr_dog_000045.jpg  
                    ...
                /wild
                    /flickr_wild_000004.jpg 
                    /flickr_wild_000012.jpg
                    ...
        /IMAGENET10:
            /train:
                    /n02056570
                        /n02056570_41.JPEG
                        /n02056570_42.JPEG
                        ...
                    /n02085936
                        /n02085936_14.JPEG
                        /n02085936_26.JPEG
                        ...
                    /n02128757
                        /n02128757_10.JPEG
                        /n02128757_34.JPEG
                        ...
                    ...
            /val:
                    /n02056570
                        /n02056570_45.JPEG
                        /n02056570_48.JPEG
                    /n02085936
                        /n02085936_37.JPEG
                        /n02085936_37.JPEG
                        ...
                    /n02128757
                        /n02128757_114.JPEG
                        /n02128757_226.JPEG
                        ...
                    ...
    ```
</details>

---

## 🏋️ 8. Training

SITP is trained in two stages:

- **Stage 1:** train the semantic communication backbone over multiple SNR conditions;
- **Stage 2:** fine-tune the model under packet losses with semantic feature interleaving.

### Stage 1: Semantic Communication Backbone

```bash
CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch \
  --nproc_per_node=2 train.py \
  --training \
  --train-stage=1 \
  --trainset='IMAGENET10' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --C=96 \
  --multiple-snr='2,4,6,8,10,12,14' \
  --batch-size=52
```

### Stage 2: Interleaving-Aware Fine-Tuning

```
CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch \
  --nproc_per_node=2 train.py \
  --training \
  --train-stage=2 \
  --trainset='IMAGENET10' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='iid' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=56 \
  --frame-len=256 \
  --interleave-mode='random' \
  --frame-loss-rate='0.0,0.1,0.2,0.3,0.4,0.5'
```

**NOTE: Training logs, checkpoints, reconstructed samples, and figures are automatically saved under the timestamped ./history directory.**

---



## 🧪 Evaluation

### Stage 1: Backbone Evaluation

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test.py \
  --train-stage=1 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=256
```

### Stage 2: IID Packet-Loss Evaluation

#### Sequential interleaving

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='iid' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=56 \
  --frame-len=1024 \
  --interleave-mode='sequential' \
  --frame-loss-rate='0.0,0.1,0.2,0.3,0.4,0.5'
```

#### Random interleaving

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='iid' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=56 \
  --frame-len=128 \
  --interleave-mode='random' \
  --frame-loss-rate='0.0,0.1,0.2,0.3,0.4,0.5'
```

### Protocol Comparison

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test_protocol.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='iid' \
  --protocol='SITP' \
  --C=96 \
  --batch-size=256 \
  --frame-len=1024 \
  --cita=2 \
  --interleave-mode='sequential'
```

### Gilbert-Elliott Burst-Loss Evaluation

#### Sequential interleaving

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='ge' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=1 \
  --frame-len=256 \
  --interleave-mode='sequential' \
  --frame-loss-rate='0.0,0.1,0.2,0.3,0.4,0.5'
```

#### Random interleaving

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='ge' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=1 \
  --frame-len=256 \
  --interleave-mode='random' \
  --frame-loss-rate='0.6,0.65,0.7,0.75,0.8,0.85'
```

### Fixed Burst-Length Evaluation

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 test_protocol.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='burst' \
  --protocol='SITP' \
  --C=96 \
  --batch-size=4 \
  --frame-len=256 \
  --interleave-mode='sequential' \
  --burst-pkts=300
```

### Reconstruction Visualization

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master-port=23456 test_save_image.py \
  --train-stage=2 \
  --trainset='AFHQ' \
  --distortion-metric='MSE' \
  --channel-type='awgn' \
  --frame-loss-type='iid' \
  --protocol='SITP' \
  --C=96 \
  --multiple-snr='10' \
  --batch-size=4 \
  --frame-len=256 \
  --interleave-mode='sequential'
```

---

## 📥 Pretrained Models

Pretrained models are available from:

- [Baidu Netdisk](https://pan.baidu.com/s/13_Lb8wFVio9PFU4jiySrhA) — extraction code: `hxzo`
- [Google Drive](https://drive.google.com/drive/folders/1YdnShbfIT03p_e30vjkV2wPKYOQPmUWp?usp=share_link)

Please place the downloaded checkpoints in the directory expected by the corresponding training or evaluation script.

---

## 📊 Results

SITP is designed to provide:

- lower latency than TCP by removing handshake and retransmission;
- better reconstruction quality than UDP by retaining corrupted semantic payloads;
- improved robustness under IID and burst packet losses;
- additional protection through cross-image semantic feature interleaving.

Detailed quantitative and qualitative results are reported in the paper.

---

## 📝 Citation

Please cite our paper when using this repository:

```bibtex
@article{wang2026sitp,
  title   = {SITP: A High-Reliability Semantic Information Transport Protocol Without Retransmission for Semantic Communication},
  author  = {Wang, Yunhao and Ma, Shuai and Wu, Youlong and Shi, Guangming and Cheng, Xiang and Liu, Yuxuan and He, Pengfei},
  journal = {IEEE Transactions on Communications},
  year    = {2026}
}
```

---

## 📄 License

This project is released under the license provided in the [`LICENSE`](./LICENSE) file.

---

## 📬 Contact

For questions or discussions, please contact:

**Yunhao Wang**  
Peking University  
Email: `yunhaowang@stu.pku.edu.cn`

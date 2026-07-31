
<h1 align="center">
** SITP **
</h1>


<p align="center">

**A High-Reliability Semantic Information Transport Protocol Without Retransmission for Semantic Communication**

IEEE Transactions on Communications, 2026

</p>


<p align="center">

<a href="https://ieeexplore.ieee.org/document/11517511">
<img src="https://img.shields.io/badge/Paper-IEEE%20TCOM-blue">
</a>

<a href="https://github.com/WYHxuebi/SITP">
<img src="https://img.shields.io/github/stars/WYHxuebi/SITP?style=social">
</a>

arxiv
dataset
weight
license
contributors
reposize
</p>



## 📌 Overview


Traditional communication protocols, such as TCP and UDP, are designed for
bit-level reliability. However, semantic communication (SemCom) systems can
tolerate noisy semantic representations and reconstruct meaningful information
from corrupted features.

To bridge this gap, we propose **Semantic Information Transport Protocol
(SITP)**, a novel transport-layer protocol specifically designed for semantic
communication.


Unlike TCP, SITP removes handshake and retransmission mechanisms to achieve
low latency.

Unlike UDP, SITP does not discard corrupted packets. Instead, SITP verifies
only packet headers and preserves noisy payloads for semantic decoding.


The proposed SITP achieves:

- ✅ TCP-level reliability
- ✅ UDP-level latency
- ✅ Robust semantic reconstruction under packet errors
- ✅ Enhanced robustness against burst packet losses



## 🚀 Key Contributions


### 1. Semantic Information Transport Protocol (SITP)

SITP introduces a new transport-layer paradigm for semantic communication.

The key idea is:

> "Verify the header, preserve the payload."

The SITP packet only performs checksum verification on header information,
while allowing corrupted semantic payloads to be delivered to the semantic
decoder.

Compared with conventional protocols:

| Protocol | Retransmission | Payload Verification | Semantic Payload Preservation |
|---|---|---|---|
| TCP | ✓ | Full | ✗ |
| UDP | ✗ | Full | ✗ |
| UDP-Lite | ✗ | Partial | ✗ |
| **SITP (Ours)** | ✗ | Header-only | ✓ |



---

### 2. Cross-layer Packet Loss Modeling


SITP establishes a cross-layer analytical model considering:

- Physical layer
- Data-link layer
- Network layer
- Transport layer
- Application layer


The model provides a unified relationship between:



# 训练流程

## Train

### stage 1

#### 单卡，单信噪比
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 train.py --training --train-stage=1 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=48

#### 多卡，单信噪比
CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch --nproc_per_node=2 train.py --training --train-stage=1 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=48

#### 多卡，多信噪比
CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch --nproc_per_node=2 train.py --training --train-stage=1 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='2, 4, 6, 8, 10, 12, 14' --batch-size=52

### stage 2
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 train.py --training --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --C=96 --multiple-snr='10' --batch-size=56 --frame-len=256 --interleave-mode='random' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'


CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch --nproc_per_node=2 train.py --training --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=56 --frame-len=256 --interleave-mode='sequential' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch --nproc_per_node=2 train.py --training --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=56 --frame-len=256 --interleave-mode='random' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch --nproc_per_node=2 train.py --training --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --C=96 --multiple-snr='10' --batch-size=56 --frame-len=256 --interleave-mode='random' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

## Test

### stage 1

#### 单卡，单信噪比
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=1 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=256

### stage 2

#### 单卡，SE，多丢包率
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --C=96 --multiple-snr='10' --batch-size=56 --frame-len=1024 --interleave-mode='sequential' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

#### 单卡，RA，多丢包率
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --C=96 --multiple-snr='10' --batch-size=56 --frame-len=128 --interleave-mode='random' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

#### 单卡，SE，协议对比
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test_protocol.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='QSTP' --C=96 --batch-size=256 --frame-len=1024 --cita=2 --interleave-mode='sequential'




CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=64 --frame-len=128 --interleave-mode='sequential' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --C=96 --multiple-snr='10' --batch-size=64 --frame-len=128 --interleave-mode='random' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='ge' --C=96 --multiple-snr='10' --batch-size=1 --frame-len=256 --interleave-mode='sequential' --frame-loss-rate='0.0, 0.1, 0.2, 0.3, 0.4, 0.5'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='ge' --C=96 --multiple-snr='10' --batch-size=1 --frame-len=256 --interleave-mode='random' --frame-loss-rate='0.6, 0.65, 0.7, 0.75, 0.8, 0.85'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test_protocol.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='SITP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='sequential' --burst-pkts=300



CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master-port=23456 test_save_image.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='SITP' --C=96 --multiple-snr='10' --batch-size=4 --frame-len=256 --interleave-mode='sequential'

## 预训练权重
All pretrain models can be found in [Baidu netdisk](https://pan.baidu.com/s/13_Lb8wFVio9PFU4jiySrhA "password:hxzo")(password:hxzo) or [Google drive](https://drive.google.com/drive/folders/1YdnShbfIT03p_e30vjkV2wPKYOQPmUWp?usp=share_link).


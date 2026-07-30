#!/usr/bin/env bash
# Burst Channel
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='SITP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='random' --burst-pkts=300

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='SITP' --C=96 --batch-size=8 --frame-len=256 --interleave-mode='random' --burst-pkts=300

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='SITP' --C=96 --batch-size=16 --frame-len=256 --interleave-mode='random' --burst-pkts=300

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='SITP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='sequential' --burst-pkts=300


CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='TCP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='random' --burst-pkts=300

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='TCP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='sequential' --burst-pkts=300


CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='UDP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='random' --burst-pkts=300

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='AFHQ' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='burst' --protocol='UDP' --C=96 --batch-size=4 --frame-len=256 --interleave-mode='sequential' --burst-pkts=300

echo "All done."
#!/usr/bin/env bash
# AWGN Channel
set -euo pipefail


CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='SITP' --C=96 --batch-size=256 --frame-len=256 --interleave-mode='sequential'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='TCP' --C=96 --batch-size=256 --frame-len=256 --interleave-mode='sequential'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='UDP' --C=96 --batch-size=256 --frame-len=256 --interleave-mode='sequential'


CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='SITP' --C=96 --batch-size=256 --frame-len=512 --interleave-mode='sequential'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='TCP' --C=96 --batch-size=256 --frame-len=512 --interleave-mode='sequential'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='UDP' --C=96 --batch-size=256 --frame-len=512 --interleave-mode='sequential'


CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='SITP' --C=96 --batch-size=256 --frame-len=1024 --interleave-mode='sequential'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='TCP' --C=96 --batch-size=256 --frame-len=1024 --interleave-mode='sequential'

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 test.py --train-stage=2 --trainset='IMAGENET10' --distortion-metric='MSE' --channel-type='awgn' --frame-loss-type='iid' --protocol='UDP' --C=96 --batch-size=256 --frame-len=1024 --interleave-mode='sequential'

echo "All done."
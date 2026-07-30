from net.network_protocol import SITP_Protocol
from net.frame import fl_rate_calu
from data.datasets import get_loader
from loss.distortion import *
from utils import *
import torch
import torch.nn as nn
import torch.distributed
import os
from datetime import datetime
import argparse
import time


# Hyperparameters
parser = argparse.ArgumentParser(description='SITP_Protocol')
parser.add_argument('--train-stage', type=int, default=2, choices=[1, 2])
parser.add_argument('--trainset', type=str, default='AFHQ',
                    choices=['CIFAR10', 'AFHQ', 'IMAGENET10'])
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'])
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'])
parser.add_argument('--frame-loss-type', type=str, default='iid', choices=['iid', 'ge', 'burst'])
parser.add_argument('--protocol', type=str, default='SITP', choices=['TCP', 'UDP', 'SITP'])
parser.add_argument('--C', type=int, default=96)
parser.add_argument('--batch-size', type=int, default=56)
parser.add_argument('--frame-len', type=int, default=1024)
parser.add_argument('--interleave-mode', type=str, default='sequential')
parser.add_argument('--burst-pkts', type=int, default=24*22)
parser.add_argument('--local-rank', type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
args = parser.parse_args()


# Related Configuration
class config():

    # System Settings
    seed = 0
    pass_channel = True
    device = torch.device("cuda", args.local_rank)
    CUDA = True
    norm = False
    lpips = True
    channel = args.frame_loss_type
    isTrain = False

    SNR_db = np.arange(7, 15, 0.3)
    protocol = args.protocol

    # Packet Loss Rates
    frame_loss_rates = [0.0]

    # Log Settings
    print_step = 25
    plot_step = 10000
    timename = str(datetime.now().__str__()[:-7]).replace(" ", "__").replace(":", "_")
    suffixname = "ADAPTIVE_FLR"
    argsname = "_STAGE{}_val_{}_".format(args.train_stage, args.interleave_mode[:2].upper()) + suffixname + f"_{args.trainset}_{channel.upper()}{args.burst_pkts}_FL{args.frame_len}_G{args.batch_size}_{args.protocol.upper()}"
    filename = timename + argsname
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log{}.log'.format(argsname)
    samples = workdir + '/samples'
    models = workdir + '/models'
    pictures = workdir + '/pictures'
    logger = None

    # CIFAR10
    if args.trainset == 'CIFAR10':
        save_model_freq = 20
        image_dims = (3, 32, 32)
        train_data_dir = "./dataset/CIFAR10" 
        test_data_dir = "./dataset/CIFAR10"
        batch_size = args.batch_size
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True)
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm)
        
    # AFHQ
    elif args.trainset == 'AFHQ':
        save_model_freq = 20
        image_dims = (3, 256, 256)
        train_data_dir = ["./dataset/AFHQ/train"]
        test_data_dir = ["./dataset/AFHQ/val"]
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

    # ImageNet-10
    elif args.trainset == 'IMAGENET10':
        save_model_freq = 10
        image_dims = (3, 256, 256)
        train_data_dir = ["./dataset/IMAGENET10/train"]
        test_data_dir = ["./dataset/IMAGENET10/test"]
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


# Weight Loading
def load_weights(model, model_path):
    pretrained = torch.load(model_path, map_location = torch.device("cuda", args.local_rank))
    model_dict = model.module.state_dict()
    pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.module.load_state_dict(model_dict, strict=True)
    del pretrained


# Test Program
def test(args):

    # Test Mode
    net.eval()
    net.module.config.isTrain = False
    
    # Relevant Metrics
    elapsed, cbrs, snrs = [AverageMeter() for _ in range(3)]
    psnrs, msssims, lpipses = [AverageMeter() for _ in range(3)]
    metrics = [elapsed, cbrs, snrs, psnrs, msssims, lpipses]

    # Multiple SNRs
    multiple_snr = config.SNR_db

    # Evaluation Metrics
    results_psnr = np.zeros((len(multiple_snr)))
    results_msssim = np.zeros((len(multiple_snr)))
    results_lpips = np.zeros((len(multiple_snr)))
    results_psnr_w50  = np.zeros((len(multiple_snr)))

    # Test
    for idx_snr, SNR in enumerate(multiple_snr):
            
        psnr_samples = []

        # Calculate Packet Loss Rate
        frame_loss_rate = fl_rate_calu(config.protocol, M=16, snr=SNR, 
                                       frame_len=args.frame_len)

        # Set Packet Loss Rate
        net.module.FrameInterleaver.loss_rate = frame_loss_rate
        net.module.FrameInterleaver.p_b = frame_loss_rate
        net.module.FrameInterleaver.burst_pkts = args.burst_pkts

        with torch.no_grad():
            for _, (input, _) in enumerate(test_loader):
                
                # Start Time
                start_time = time.time()

                # Load in Cuda
                input = input.cuda(args.local_rank)

                # Forward Propagation
                Tx_features = net.module.encoding(input, SNR)
                Tx_features = torch.round(Tx_features)
                Tx_bits = net.module.tensor_to_binary_tensor(Tx_features)
                Tx_frames, padded, frame_pad_len = net.module.frame_interleave(Tx_bits)
                Tx_symbols = net.module.qam16_modulate(Tx_frames)
                Rx_symbols = net.module.add_nosiychannel(Tx_symbols, SNR)
                Rx_frames = net.module.qam16_demodulate(Rx_symbols)
                Rx_frames, _ = net.module.frame_loss(Rx_frames)
                Rx_bits = net.module.frame_deinterleave(Rx_frames, padded, frame_pad_len, Tx_bits.shape)
                Rx_features = net.module.binary_tensor_to_tensor(Rx_bits).to(torch.float)
                recon_image, CBR, SNR, mse, lpips, _ = net.module.decoding_mask(Rx_features, 
                                                                                     Tx_features, 
                                                                                     input, 
                                                                                     SNR, 
                                                                                     mask_flag=False)

                # PSNR
                x = input.clamp(0., 1.) * 255.0
                y = recon_image.clamp(0., 1.) * 255.0
                mse_per = (x - y).pow(2).flatten(1).mean(1).clamp_min(1e-12)
                psnr_per = 10.0 * torch.log10((255.0**2) / mse_per)
                B = x.size(0)
                
                # Evaluation Records
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)

                # PSNR / MS-SSIM / LPIPS
                if mse.item() > 0:
                    psnrs.update(psnr_per.mean().item(), n=B)
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                    lpipses.update(lpips)
                else:
                    psnrs.update(100)
                    msssims.update(1.0)
                    lpipses.update(0.0)

                # Sample Records
                psnr_samples.extend(psnr_per.detach().cpu().tolist())

                # Log
                if args.local_rank == 0:
                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'Loss_rate {net.module.FrameInterleaver.loss_rate:.2f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'LPIPS {lpipses.val:.3f} ({lpipses.avg:.3f})']))
                    logger.info(log)

        results_psnr[idx_snr] = psnrs.avg
        results_msssim[idx_snr] = msssims.avg
        results_lpips[idx_snr] = lpipses.avg
        results_psnr_w50[idx_snr] = bottom_k_mean_from_list(psnr_samples, k=100)

        for t in metrics:
            t.clear()

    # Printing Related Metrics
    if args.local_rank == 0:
        log = (' | '.join([
            f'PSNR {results_psnr.tolist()})\n',
            f'MSSSIM {results_msssim.tolist()})\n',
            f'LPIPS {results_lpips.tolist()})\n',
            f'PSNR Worst-50 mean {results_psnr_w50.tolist()})\n',]))
        logger.info(log)

    return results_psnr, results_msssim, results_lpips, results_psnr_w50


# Main Program
if __name__ == '__main__':

    # Args Parameters
    print("args:", args)

    # Number of GPUs
    num_GPU = torch.cuda.device_count()
    print(f"Find {num_GPU} GPUs!")

    # Initialize Process Group
    torch.distributed.init_process_group("nccl", world_size=num_GPU, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)

    # Initialize Seeds
    seed_torch()
    torch.manual_seed(seed=config.seed)

    # Configuration Logs
    logger = None
    if args.local_rank == 0:
        logger = logger_configuration(config, save_log=True)
        logger.info(args)

    # Instantiate Network
    net = SITP_Protocol(args, config).cuda(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # MS-SSIM Instantiation
    if args.trainset == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda(args.local_rank)
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=5, channel=3).cuda(args.local_rank)

    # Loading Network Weights
    if args.train_stage == 1:
        model_path = f"./weight/{args.trainset}_IW_AWGN.model"
    elif args.train_stage == 2:
        model_path = f"./weight/{args.trainset}_IW_AWGN_ADAPTIVE_FLR.model"
    else:
        raise ValueError(f"Unkonwn train stage {args.train_stage}!")
    load_weights(net, model_path)

    # Dataset
    _, test_dataset = get_loader(args, config)

    # Dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, pin_memory=True, 
                                              batch_size=args.batch_size, 
                                              drop_last=True, shuffle=False)

    # Test
    results_psnr, results_msssim, results_lpips, results_psnr_w50 = test(args)

    # Saving Metrics
    if args.local_rank == 0:
        np.save(config.samples + '/PSNR.npy', results_psnr)
        np.save(config.samples + '/SSIM.npy', results_msssim)
        np.save(config.samples + '/LPIPS.npy', results_lpips)
        np.save(config.samples + '/PSNR_WORST50.npy', results_psnr_w50)
from net.network import SITP
from data.datasets import get_loader
from loss.distortion import *
from utils import *
import torch
import torch.nn as nn
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
import argparse
from trainer import train


# Hyperparameters
parser = argparse.ArgumentParser(description='SITP')
parser.add_argument('--train-stage', type=int, default=1, 
                    choices=[1, 2])
parser.add_argument('--trainset', type=str, default='IMAGENET10', 
                    choices=['CIFAR10', 'AFHQ', 'IMAGENET10'])
parser.add_argument('--distortion-metric', type=str, default='MSE', 
                    choices=['MSE', 'MS-SSIM'])
parser.add_argument('--channel-type', type=str, default='awgn', 
                    choices=['awgn', 'rayleigh'])
parser.add_argument('--frame-loss-type', type=str, default='iid', 
                    choices=['iid', 'ge', 'burst'])
parser.add_argument('--C', type=int, default=96)
parser.add_argument('--multiple-snr', type=str, 
                    default='-5, 0, 5, 10, 15, 20, 25')
parser.add_argument('--batch-size', type=int, default=56)
parser.add_argument('--frame-len', type=int, default=256)
parser.add_argument('--interleave-mode', type=str, default='sequential')
parser.add_argument('--frame-loss-rate', type=str, default='0.0')
parser.add_argument('--local-rank', type=int, 
                    default=int(os.environ.get("LOCAL_RANK", 0)))
args = parser.parse_args()


# Related configuration
class config():

    # System Settings
    seed = 0
    device = torch.device("cuda", args.local_rank)
    pass_channel = True
    norm = False
    isTrain = True
    channel = args.frame_loss_type
    bits = 4

    # Multiple SNRs
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])

    # Multiple Packet Loss Rates
    if args.train_stage == 1:
        frame_loss_rates = [0.0]
    elif args.train_stage == 2:
        frame_loss_rates = args.frame_loss_rate.split(",")
        for i in range(len(frame_loss_rates)):
            frame_loss_rates[i] = float(frame_loss_rates[i])
    else:
        raise ValueError(f"Unkonwn train stage {args.train_stage}!")

    # Log Settings
    print_step = 25
    plot_step = 10000
    timename = str(datetime.now().__str__()[:-7]).replace(" ", "__").replace(":", "_")
    argsname = "_STAGE{}".format(args.train_stage)
    filename = timename + argsname
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log{}.log'.format(argsname)
    samples = workdir + '/samples'
    models = workdir + '/models'
    pictures = workdir + '/pictures'
    logger = None

    # Training Epochs
    JZfinetune_epoch = 20           # Num. of Soft Quantization Training Epochs (Stage 1)
    DECfinetune_alpha_epoch = 80    # Num. of Decoder Fine-tuning Epochs (Stage 2.1-A)
    ENCfinetune_epoch = 20          # Num. of Codec Fine-tuning Epochs (Stage 2.1-B + 2.2-B)
    DECfinetune_beta_epoch = 20     # Num. of Decoder Fine-tuning Epochs (Stage 2.2-A)
    DECunleash_epoch = 20           # Num. of Codec Fine-tuning Epochs (Stage 3)
    FLfinetune_epoch = 40           # Num. of Packet Loss Rate Fine-tuning Epochs (Stage 4)

    # CIFAR10
    if args.trainset == 'CIFAR10':
        save_model_freq = 20
        image_dims = (3, 32, 32)                                                     # CIFAR10 图像大小
        train_data_dir = "./dataset/CIFAR10"                                         # 训练数据根路径
        test_data_dir = "./dataset/CIFAR10"                                          # 测试数据根路径
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
        save_model_freq = 10
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
    net = SITP(args, config)
    net = nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank])

    # MS-SSIM Instantiation
    if args.trainset == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda(args.local_rank)
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=5, channel=3).cuda(args.local_rank)

    # Dataset
    train_dataset, test_dataset = get_loader(args, config)
    train_sampler = DistributedSampler(train_dataset, seed=1)
    if args.local_rank == 0:
        logger.info(f"train_dataset: {len(train_dataset)}")
        logger.info(f"test_dataset: {len(test_dataset)}")

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, pin_memory=True, 
                                               batch_size=args.batch_size, drop_last=True, 
                                               sampler=train_sampler, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, pin_memory=True, 
                                              batch_size=args.batch_size, 
                                              drop_last=True, shuffle=False)

    # Step
    global_step = 0

    # Train Stage 1
    if args.train_stage == 1:

        # Stage 1
        # JZfinetune
        lr = 1e-4
        min_lr = 1e-6
        total_epoches = config.JZfinetune_epoch
        stage_name = "JZfinetune"
        model_path = f"./weight/{args.trainset}_IW_AWGN.model"
        save_best_path = config.models +"/DSC_best_model_JZfinetune.model"
        load_weights(net, model_path)
        print ("Starting to finetune model with JZfinetune...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Stage 2.1
        # Finetuning actor (raw UNet+decoder+mask atk)
        lr = 1e-4
        min_lr = 1e-6
        total_epoches = config.DECfinetune_alpha_epoch
        stage_name = "DECfinetune"
        model_path = config.models + "/DSC_best_model_JZfinetune.model"
        save_best_path = config.models +"/DSC_best_model_DECfinetune_1st.model"
        print ("Starting to finetune actor with raw U-Net (1st) and Mask ATK...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Finetune environment (encoder+mask atk)
        lr = 1e-4
        min_lr = 1e-6
        total_epoches = config.ENCfinetune_epoch
        stage_name = "ENCfinetune"
        model_path = config.models + "/DSC_best_model_DECfinetune_1st.model"
        save_best_path = config.models +"/DSC_best_model_ENCfinetune_1st.model"
        print ("Starting to finetune environment (1st) with Mask ATK...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Stage 2.2
        # Finetune actor (UNet+decoder+mask atk)
        lr = 1e-4
        min_lr = 1e-6
        total_epoches = config.DECfinetune_beta_epoch
        stage_name = "DECfinetune"
        model_path = config.models + "/DSC_best_model_ENCfinetune_1st.model"
        save_best_path = config.models +"/DSC_best_model_DECfinetune_2nd.model"
        print ("Starting to finetune actor (2nd) with Mask ATK...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Finetune environment (encoder+mask atk)
        lr = 1e-4
        min_lr = 1e-6
        total_epoches = config.ENCfinetune_epoch
        stage_name = "ENCfinetune"
        model_path = config.models + "/DSC_best_model_DECfinetune_2nd.model"
        save_best_path = config.models +"/DSC_best_model_ENCfinetune_2nd.model"
        print ("Starting to finetune environment (2nd) with Mask ATK...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Stage 2.3
        # Finetune actor (UNet+decoder+mask atk)
        lr = 1.5e-5
        min_lr = 1e-6
        total_epoches = config.DECfinetune_beta_epoch
        stage_name = "DECfinetune"
        model_path = config.models + "/DSC_best_model_ENCfinetune_2nd.model"
        save_best_path = config.models +"/DSC_best_model_DECfinetune_3rd.model"
        print ("Starting to finetune actor (3rd) with Mask ATK...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Finetune environment (encoder+mask atk)
        lr = 1.5e-5
        min_lr = 1e-6
        total_epoches = config.ENCfinetune_epoch
        stage_name = "ENCfinetune"
        model_path = config.models + "/DSC_best_model_DECfinetune_3rd.model"
        save_best_path = config.models +"/DSC_best_model_ENCfinetune_3rd.model"
        print ("Starting to finetune environment (3rd) with Mask ATK...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)

        # Stage 3
        # Finetune actor (UNet+decoder+mask atk)
        lr = 1.5e-5
        min_lr = 1e-6
        total_epoches = config.DECunleash_epoch
        stage_name = "DECunleash"
        model_path = config.models + "/DSC_best_model_ENCfinetune_3rd.model"
        save_best_path = config.models +"/DSC_best_model_DECunleash.model"
        print ("Starting to finetune actor (4th)...")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)
    
    # Train Stage 2
    elif args.train_stage == 2:
        lr = 1e-4
        min_lr = 1e-6
        total_epoches = config.FLfinetune_epoch
        stage_name = "FLfinetune_epoch"
        model_path = f"./weight/{args.trainset}_IW_AWGN.model"
        save_best_path = config.models +"/DSC_best_model_FLfinetune.model"
        load_weights(net, model_path)
        print ("Starting to finetune FL")
        train(net, args, config, total_epoches, train_loader, test_loader,
                lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step)
    
    else:
        raise ValueError(f"Unknown train stage: {args.train_stage}")
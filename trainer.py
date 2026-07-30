import numpy as np
import torch.optim as optim
import time
from random import choice
from loss.distortion import *
from utils import *


# Single Epoch Training
def train_one_epoch(net, args, config, epoch, optimizer, 
                    train_loader, stage_name, CalcuSSIM, logger, global_step):

    # Relevant Metrics
    elapsed, losses, cbrs, snrs = [AverageMeter() for _ in range(4)]
    psnrs, msssims, lpipses = [AverageMeter() for _ in range(3)]
    metrics = [elapsed, losses, cbrs, snrs, psnrs, msssims, lpipses]

    # Multiple SNRs
    multiple_snr = config.multiple_snr
        
    # Multiple Packet Loss Rates
    frame_loss_rates = config.frame_loss_rates

    for batch_idx, (input, label) in enumerate(train_loader):

        # Gradient Zeroing
        optimizer.zero_grad()

        # Start Time
        start_time = time.time()

        # Total Step
        global_step += 1

        # Load in Cuda
        input = input.cuda(args.local_rank)

        # Random SNR
        chan_param = choice(multiple_snr)
        
        # Random Packet Loss Rate
        frame_loss_rate = choice(frame_loss_rates)
        net.module.FrameInterleaver.loss_rate = frame_loss_rate
        
        # Forward Propagation
        # Training Stage 1
        if args.train_stage == 1:
            if stage_name == 'JZfinetune':
                Tx_features = net.module.encoding(input, chan_param)
                Rx_features = Tx_features + 0.5 * torch.randn_like(Tx_features)
                recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_native(Rx_features, 
                                                                                       input, 
                                                                                       chan_param)

            elif stage_name == 'DECfinetune':
                with torch.no_grad():
                    Tx_features = net.module.encoding(input, chan_param)
                    Tx_features = torch.round(Tx_features)
                    Tx_bits = net.module.tensor_to_binary_tensor(Tx_features)
                    Tx_symbols = net.module.qam16_modulate(Tx_bits)
                    Rx_symbols = net.module.add_nosiychannel(Tx_symbols, chan_param)
                    Rx_bits = net.module.qam16_demodulate(Rx_symbols)
                    Rx_features = net.module.binary_tensor_to_tensor(Rx_bits).to(torch.float)
                recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                     Tx_features, 
                                                                                     input, 
                                                                                     chan_param,
                                                                                     mask_flag=True)

            elif stage_name == 'ENCfinetune':
                Tx_features = net.module.encoding(input, chan_param)
                Rx_features = Tx_features + 0.5 * torch.randn_like(Tx_features)
                recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                     Tx_features, 
                                                                                     input, 
                                                                                     chan_param, 
                                                                                     mask_flag=True)

            elif stage_name == 'DECunleash':
                with torch.no_grad():
                    Tx_features = net.module.encoding(input, chan_param)
                    Tx_features = torch.round(Tx_features)
                    Tx_bits = net.module.tensor_to_binary_tensor(Tx_features)
                    Tx_symbols = net.module.qam16_modulate(Tx_bits)
                    Rx_symbols = net.module.add_nosiychannel(Tx_symbols, chan_param)
                    Rx_bits = net.module.qam16_demodulate(Rx_symbols)
                    Rx_features = net.module.binary_tensor_to_tensor(Rx_bits).to(torch.float)
                recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                     Tx_features, 
                                                                                     input, chan_param, 
                                                                                     mask_flag=False)
            else:
                raise ValueError(f"Unknown the stage: {stage_name}")
        
        # Training Stage 2
        elif args.train_stage == 2:
            with torch.no_grad():
                Tx_features = net.module.encoding(input, chan_param)
                Tx_features = torch.round(Tx_features)
                Tx_bits = net.module.tensor_to_binary_tensor(Tx_features)
                Tx_frames, padded, frame_pad_len = net.module.frame_interleave(Tx_bits)
                Tx_symbols = net.module.qam16_modulate(Tx_frames)
                Rx_symbols = net.module.add_nosiychannel(Tx_symbols, chan_param)
                Rx_frames = net.module.qam16_demodulate(Rx_symbols)
                Rx_frames, _ = net.module.frame_loss(Rx_frames)
                Rx_bits = net.module.frame_deinterleave(Rx_frames, padded, frame_pad_len, Tx_bits.shape)
                Rx_features = net.module.binary_tensor_to_tensor(Rx_bits).to(torch.float)
            recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                 Tx_features, 
                                                                                 input, 
                                                                                 chan_param, 
                                                                                 mask_flag=False)
        else:
            raise ValueError(f"Unknown train stage: {args.train_stage}")
        
        # Calculate Total Loss
        loss = loss_G

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Evaluation Records
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)

        # PSNR / MS-SSIM / LPIPS
        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)
            lpipses.update(lpips)
        else:
            psnrs.update(100)
            msssims.update(1.0)
            lpipses.update(0.0)

        # Log
        if args.local_rank == 0:
            learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
            if (global_step % config.print_step) == 0:
                process = (global_step % (train_loader.__len__())) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'{stage_name}',
                    f'Epoch {epoch}',
                    f'Step [{global_step % (train_loader.__len__())}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'Loss_rate {net.module.FrameInterleaver.loss_rate:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'LPIPS {lpipses.val:.3f} ({lpipses.avg:.3f})',
                    f'Lr {learning_rate}']))
                logger.info(log)

    for i in metrics:
        i.clear()

    return losses.avg, global_step


# Test
def test(net, args, test_loader, stage_name, CalcuSSIM, logger):

    # Relevant Metrics
    elapsed, cbrs, snrs = [AverageMeter() for _ in range(3)]
    psnrs, msssims, lpipses, losses = [AverageMeter() for _ in range(4)]
    metrics = [elapsed, cbrs, snrs, psnrs, msssims, lpipses]

    # Multiple SNRs
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])

    # Evaluation Metrics
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    results_lpips = np.zeros(len(multiple_snr))

    # Test
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(test_loader):

                # Start Time
                start_time = time.time()

                # Load in Cuda
                input = input.cuda(args.local_rank)

                # Forward Propagation
                # Train Stage 1
                if args.train_stage == 1:
                    if stage_name == 'JZfinetune':
                        Tx_features = net.module.encoding(input, SNR)
                        Rx_features = Tx_features + 0.5 * torch.randn_like(Tx_features)
                        recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_native(Rx_features, 
                                                                                               input, 
                                                                                               SNR)

                    elif stage_name == 'DECfinetune':
                        Tx_features = net.module.encoding(input, SNR)
                        Tx_features = torch.round(Tx_features)
                        Tx_bits = net.module.tensor_to_binary_tensor(Tx_features)
                        Tx_symbols = net.module.qam16_modulate(Tx_bits)
                        Rx_symbols = net.module.add_nosiychannel(Tx_symbols, SNR)
                        Rx_bits = net.module.qam16_demodulate(Rx_symbols)
                        Rx_features = net.module.binary_tensor_to_tensor(Rx_bits).to(torch.float)
                        recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                             Tx_features, 
                                                                                             input, 
                                                                                             SNR, 
                                                                                             mask_flag=True)

                    elif stage_name == 'ENCfinetune':
                        Tx_features = net.module.encoding(input, SNR)
                        Rx_features = Tx_features + 0.5 * torch.randn_like(Tx_features)
                        recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                             Tx_features, 
                                                                                             input, 
                                                                                             SNR, 
                                                                                             mask_flag=True)

                    elif stage_name == 'DECunleash':
                        Tx_features = net.module.encoding(input, SNR)
                        Tx_features = torch.round(Tx_features)
                        Tx_bits = net.module.tensor_to_binary_tensor(Tx_features)
                        Tx_symbols = net.module.qam16_modulate(Tx_bits)
                        Rx_symbols = net.module.add_nosiychannel(Tx_symbols, SNR)
                        Rx_bits = net.module.qam16_demodulate(Rx_symbols)
                        Rx_features = net.module.binary_tensor_to_tensor(Rx_bits).to(torch.float)
                        recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                             Tx_features, 
                                                                                             input, 
                                                                                             SNR, 
                                                                                             mask_flag=False)
                    else:
                        raise ValueError(f"Unknown the stage: {stage_name}")
                
                # Train Stage 2
                elif args.train_stage == 2:
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
                    recon_image, CBR, SNR, mse, lpips, loss_G = net.module.decoding_mask(Rx_features, 
                                                                                         Tx_features, 
                                                                                         input, SNR, 
                                                                                         mask_flag=False)
                else:
                    raise ValueError(f"Unknown train stage: {args.train_stage}")

                # Evaluation Records
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                losses.update(loss_G.item())

                # PSNR / MS-SSIM / LPIPS
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                    lpipses.update(lpips)
                else:
                    psnrs.update(100)
                    msssims.update(1.0)
                    lpipses.update(0.0)

                # Log
                if args.local_rank == 0:
                    log = (' | '.join([
                        f'{stage_name}',
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'Loss_rate {net.module.FrameInterleaver.loss_rate:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'LPIPS {lpipses.val:.3f} ({lpipses.avg:.3f})']))
                    logger.info(log)

        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        results_lpips[i] = lpipses.avg

        for t in metrics:
            t.clear()

    # Printing Related Metrics
    if args.local_rank == 0:
        log = (' | '.join([
            f'PSNR {results_psnr.tolist()})\n',
            f'MSSSIM {results_msssim.tolist()})\n',
            f'LPIPS {results_lpips.tolist()})\n',]))
        logger.info(log)

    return results_psnr, results_msssim, results_lpips, losses.avg


# Trainer
def train(net, args, config, total_epoches, train_loader, test_loader, 
          lr, min_lr, stage_name, CalcuSSIM, logger, save_best_path, global_step):

    # Record Loss
    loss_save = np.zeros(total_epoches)
    best_loss = 1e8

    # Define Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer.zero_grad()

    # Learning Rate Decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoches, eta_min=min_lr)

    for epoch in range(total_epoches):
        
        # Training Mode
        net.train()
        net.module.config.isTrain = True

        loss_save[epoch], global_step = train_one_epoch(net, args, config, epoch, optimizer, 
                                                        train_loader, stage_name, CalcuSSIM, 
                                                        logger, global_step)
        
        # Learning Rate Update
        scheduler.step()
        
        # Test
        if (epoch + 1) % config.save_model_freq == 0:

            # Test Mode
            net.eval()
            net.module.config.isTrain = False

            results_psnr, results_msssim, results_lpips, loss_avg = test(net, args, test_loader, stage_name, CalcuSSIM, logger)

            if args.local_rank == 0:
                torch.save(net.module.state_dict(), config.models + '/{}_{}_EP{}_IR.model'.format(stage_name, config.filename, epoch + 1))
                print("********************模型已保存********************")

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    print ("New Record Confirm, Saving Model...")
                    torch.save(net.module.state_dict(), save_best_path)
            
            # All Processes Waiting
            torch.distributed.barrier()
    
    # Saving Metrics
    if args.local_rank == 0:
        np.save(config.samples + '/LOSS.npy', loss_save)
        np.save(config.samples + '/PSNR.npy', results_psnr)
        np.save(config.samples + '/SSIM.npy', results_msssim)
        np.save(config.samples + '/LPIPS.npy', results_lpips)
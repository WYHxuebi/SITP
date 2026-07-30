import numpy as np
import torch
import random
import os
import logging
import torchvision.transforms.functional as TF


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(config, save_log=False, test_mode=False):
    """
    Log configuration
    """
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
        makedirs(config.pictures)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger

def makedirs(directory):
    """
    Create directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    """
    Save Model Parameters
    """
    torch.save(model.module.state_dict(), save_path)


def seed_torch(seed=0):
    """
    Setting Seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def concat_images(images):
    """
    Stitching Images
    """
    images = images.cpu()
    batch_size, C, H, W = images.shape

    H_concat = H * int(np.ceil(np.sqrt(batch_size)))
    W_concat = W * int(np.ceil(batch_size / np.ceil(np.sqrt(batch_size))))

    concat_image = np.zeros((C, H_concat, W_concat))

    for i in range(batch_size):
        row = i // int(np.ceil(np.sqrt(batch_size)))
        col = i % int(np.ceil(np.sqrt(batch_size)))
        concat_image[:, row*H:(row+1)*H, col*W:(col+1)*W] = images[i]
    concat_image = torch.tensor(concat_image).permute(1, 2, 0).cpu().numpy()

    return concat_image


def forzen_net(net, frozen_parameters, frozen_network, epoch, frozen_epoch, learning_rate):
    """
    Freeze Network
    """
    if frozen_network == True:
        if epoch <= frozen_epoch:
            for name, param in net.named_parameters():
                if name in frozen_parameters:
                    param.requires_grad = False
        else:
            for name, param in net.named_parameters():
                param.requires_grad = True
    else:
        for name, param in net.named_parameters():
            param.requires_grad = True
            
            
def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def save_batch_tensor_images(tensor, save_dir, prefix="img"):
    """
    Batch save CUDA Tensor images as PNGs.
    """
    os.makedirs(save_dir, exist_ok=True)
    tensor = tensor.detach().cpu()
    tensor = tensor * 255.0
    tensor = tensor.clamp(0, 255).to(torch.uint8)
    for i in range(tensor.size(0)):
        img = TF.to_pil_image(tensor[i])
        
        img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

    print(f"Saved {tensor.size(0)} images to {save_dir}")


def bottom_k_mean_from_list(values, k=50):
    """
    Returns the average of the k smallest values ​​in the list; 
    if the sample size is less than k, all values ​​are returned.
    """
    if not values:
        return float('nan')
    import numpy as np
    arr = np.asarray(values, dtype=float)
    k = min(k, arr.size)
    idx = np.argpartition(arr, k - 1)[:k]
    return float(arr[idx].mean())


def tail_metrics_single(psnr_samples, tau=24.0, worst_k=50, use_rel_tau=False):
    """
    psnr_samples: list/ndarray of PSNR (dB) for one G;
    tau: Low-quality threshold (dB); 
         when use_rel_tau=True, p10 of this distribution will be used as the threshold
    worst_k: Calculate the average of the K worst samples
    """
    x = np.asarray(psnr_samples, dtype=float)
    if x.size == 0:
        raise ValueError("psnr_samples is empty.")
    tau_used = float(np.percentile(x, 10)) if use_rel_tau else float(tau)

    mean = float(np.mean(x))
    std  = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    p1  = float(np.percentile(x, 1))
    p5  = float(np.percentile(x, 5))
    low_rate = float((x < tau_used).mean())

    k = min(int(worst_k), x.size)
    worstK = float(np.mean(np.partition(x, k-1)[:k]))

    # ES@5%
    q5 = np.percentile(x, 5)
    es5 = float(x[x <= q5].mean()) if np.any(x <= q5) else float('nan')

    return {
        "mean": mean, "std": std, "iqr": iqr, "mad": mad,
        "p1": p1, "p5": p5, "low_rate": low_rate, "ES5": es5,
        "worstK": worstK, "tau_used": tau_used, "N": int(x.size)
    }
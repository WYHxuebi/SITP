from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn


SCALE_MIN = 0.75
SCALE_MAX = 0.95

class HR_image(Dataset):

    def __init__(self, config, data_dir):
        self.imgs = []

        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*', '*.*'))
    
        self.imgs.sort()
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()


    def _transforms(self,):
        transforms_list = [
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()]
        
        return transforms.Compose(transforms_list)


    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed, idx


    def __len__(self):
        return len(self.imgs)


class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)

    def __len__(self):
        return self.len * 10


def get_loader(args, config):

    if args.trainset == 'CIFAR10':

        dataset_ = datasets.CIFAR10

        # Image Preprocessing
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        
        # Train dataset
        train_dataset = dataset_(root = config.train_data_dir,
                                 train = True,
                                 transform = transform_train,
                                 download = True)

        # Test dataset
        test_dataset = dataset_(root = config.test_data_dir,
                                train = False,
                                transform = transform_test,
                                download = True)

        train_dataset = CIFAR10(train_dataset)

    else:
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = HR_image(config, config.test_data_dir)

    return train_dataset, test_dataset


if __name__ == '__main__':

    NOCM_deep = 10
    model_mode = 2
    
    class args():
        trainset = 'AFHQ'
    trainset = args.trainset

    class config():

        seed = 0
        pass_channel = True
        CUDA = True
        device = torch.device("cuda")
        norm = False

        learning_rate = 0.0001
        tot_epoch = 45
        orthogonal = False
        user_num = 3
        angle = 30
        forzen = True

        # CIFAR10
        if trainset == 'CIFAR10':
            save_model_freq = 15
            image_dims = (3, 32, 32)
            train_data_dir = "/dataset"
            test_data_dir = "/dataset"
            batch_size = 256
            downsample = 2
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=48,
                window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True, NOC_dim=128,
                NOCM_deep = NOCM_deep, mode = model_mode
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=48,
                window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, NOC_dim=128, NOCM_deep=NOCM_deep,
                mode = model_mode
            )

        # AFHQ
        elif trainset == 'AFHQ':
            save_model_freq = 15
            image_dims = (3, 192, 192)
            train_data_dir = ["./dataset/AFHQ/train/cat", "./dataset/AFHQ/train/dog", "./dataset/AFHQ/train/wild"]
            test_data_dir = ["./dataset/AFHQ/val/cat", "./dataset/AFHQ/val/dog", "./dataset/AFHQ/val/wild"]
            batch_size = 16
            downsample = 4
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
                C=48, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True, NOC_dim=128, 
                NOCM_deep = NOCM_deep, mode = model_mode
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
                C=48, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, NOC_dim=128, NOCM_deep=NOCM_deep,
                mode = model_mode
            )

    train_dataset, test_dataset = get_loader(args, config)

    train_loader_user1 = torch.utils.data.DataLoader(dataset=train_dataset, pin_memory=True, 
                                                     batch_size=config.batch_size, drop_last=True, 
                                                     shuffle=False, num_workers=2)
    test_loader_user1 = torch.utils.data.DataLoader(dataset=test_dataset, pin_memory=True, 
                                                    batch_size=256, drop_last=True, shuffle=False)
    
    print(len(train_loader_user1))

    for batch_idx, (input_user3, _) in enumerate(train_loader_user1):
        print(input_user3.shape)
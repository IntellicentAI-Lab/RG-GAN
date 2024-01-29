from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sngan_model import ResDiscriminator32, ResGenerator32
import numpy as np
from pytorch_fid import fid_score
from pytorch_gan_metrics import get_inception_score, ImageDataset
from diffaug import DiffAugment
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def main():
    # Create the dataset
    dataset = dset.CIFAR10(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]), download=True, train=True)

    # Make sub-training dataset
    subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.data_ratio)))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)

    netD = ResDiscriminator32().to(device)
    netG = ResGenerator32(args.noise_size).to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), args.lr, (0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), args.lr, (0, 0.9))

    fixed_noise = torch.randn(64, args.noise_size, device=device)

    best_fid = 9999

    print("Starting Training Loop...")

    for epoch in range(1, args.epoch + 1):

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real_cpu = data[0].to(device)

            if diffaug_flag:
                real_cpu = DiffAugment(real_cpu, policy=policy)

            b_size = real_cpu.size(0)
            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.noise_size, device=device)
            fake = netG(noise)

            if diffaug_flag:
                fake = DiffAugment(fake, policy=policy)
            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(nn.ReLU(inplace=True)(1 + output))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, args.noise_size, device=device)
                fake = netG(noise)

                if diffaug_flag:
                    fake = DiffAugment(fake, policy=policy)

                output = netD(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                D_G_z2 = output.mean().item()

                optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%4d/%4d][%3d/%3d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Evaluation
        if epoch % args.eva_epoch == 0:
            netG.eval()

            fake = netG(fixed_noise).detach().cpu()
            torchvision.utils.save_image(torchvision.utils.make_grid(fake, padding=2, normalize=True),
                                './%s/epoch_%d.png' % (current_model_result_dir, epoch))
            print('Epoch %2d viz images had been saved!' % epoch)

            eva_size = 100
            for iii in tqdm(range(args.eva_size // eva_size), desc='Generating images'):
                Noisee = torch.randn(eva_size, args.noise_size, device=device)
                temp_fake = netG(Noisee)
                for iiii in range(eva_size):
                    torchvision.utils.save_image(temp_fake[iiii].detach(),
                                        '%s/f_%s.png' % (current_model_eva_dir, str(iii * eva_size + iiii)),
                                        normalize=True, range=(-1, 1))
            print('-' * 10 + 'Evaluation Begin' + '-' * 10)

            print('----FID----')
            print('-------------Eva FID------------')
            fid = fid_score.calculate_fid_given_paths([current_model_eva_dir, './dataset/cifar10.test.npz'],
                                                        100, device, 2048)

            print('FID : %.4f' % fid)

            if fid < best_fid:
                best_fid = fid
                print('----IS-----')
                dataset = ImageDataset(current_model_eva_dir, exts=['png', 'jpg'])
                loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=4)
                IS, IS_std = get_inception_score(loader)
                print('Inception Score: {:.2f} +/- {:.2f}'.format(IS, IS_std))

            netG.train()


if __name__ == '__main__':
    model_name = 'SNGAN'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=800)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--image_size', type=int, default=32)
    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='./dataset')
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--data_ratio', type=float, default=0.2)
    argparser.add_argument('--eva_size', type=int, default=10000)
    argparser.add_argument('--eva_epoch', type=int, default=20)
    argparser.add_argument('--diffaug', action='store_true', help='apply DiffAug')
    args = argparser.parse_args()

    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)

    current_model_result_dir = './%s/result' % model_name

    if not os.path.exists(current_model_result_dir):
        os.makedirs(current_model_result_dir)

    current_model_eva_dir = './%s/eva' % model_name

    if not os.path.exists(current_model_eva_dir):
        os.makedirs(current_model_eva_dir)

    device = "cuda"

    pre_calculated_fid_dir = './dataset/cifar10.test.npz'
    assert os.path.exists(pre_calculated_fid_dir), 'Please put pre-calculated cifar10.test.npz file into ./dataset folder, you can download it from https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC'

    if args.diffaug == 1:
        policy = 'color,translation,cutout'
        diffaug_flag = True
        print('Diffaug is now enabled')
    else:
        diffaug_flag = False

    main()
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
from rggan_model import RG_ResDiscriminator32, RG_ResGenerator32
import numpy as np
from pytorch_fid import fid_score
from pytorch_gan_metrics import get_inception_score, ImageDataset
from diffaug import DiffAugment
from tqdm import tqdm
from utils import copy_params, load_params, sparsity_regularizer, print_layer_keep_ratio, set_training_mode
from dynamic_layers import MaskedConv2d, MaskedMLP
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

    netD = RG_ResDiscriminator32().to(device)
    netG = RG_ResGenerator32(args.noise_size).to(device)

    netG_avg_param = copy_params(netG)

    netG.sparse_train_mode = True
    netD.sparse_train_mode = True

    set_training_mode(netG, netG.sparse_train_mode)
    set_training_mode(netD, netD.sparse_train_mode)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), args.lr, (0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), args.lr, (0, 0.9))

    fixed_noise = torch.randn(64, args.noise_size, device=device)

    print("Starting Training Loop...")
    best_fid = 9999
    fid_record = []

    for epoch in range(1, args.epoch + 1):

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            if diffaug_flag:
                real_cpu = DiffAugment(real_cpu, policy=policy)

            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))

            if netD.sparse_train_mode:
                sr_loss = sparsity_regularizer(netD, args.lambda_)
                errD_real = errD_real + sr_loss

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

            if netD.sparse_train_mode:
                for layer in netD.modules():
                    if isinstance(layer, MaskedConv2d) or isinstance(layer, MaskedMLP):
                        try:
                            layer.weight_orig.grad.data = layer.weight_orig.grad.data * layer.mask
                        except:
                            layer.weight.grad.data = layer.weight.grad.data * layer.mask

            optimizerD.step()

            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, args.noise_size, device=device)
                fake = netG(noise)

                if diffaug_flag:
                    fake = DiffAugment(fake, policy=policy)

                output = netD(fake).view(-1)
                errG = -torch.mean(output)

                if netG.sparse_train_mode:
                    sr_loss = sparsity_regularizer(netG, args.lambda_)
                    errG = errG + sr_loss

                errG.backward()
                D_G_z2 = output.mean().item()

                if netG.sparse_train_mode:
                    for layer in netG.modules():

                        if isinstance(layer, MaskedConv2d) or isinstance(layer, MaskedMLP):
                            try:
                                layer.weight_orig.grad.data = layer.weight_orig.grad.data * layer.mask
                            except:
                                layer.weight.grad.data = layer.weight.grad.data * layer.mask

                optimizerG.step()

                # moving average weight
                for p, avg_p in zip(netG.parameters(), netG_avg_param):
                    avg_p.mul_(0.999).add_(0.001, p.data)

            # Output training stats
            if i % 50 == 0:
                print('[%4d/%4d][%3d/%3d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Evaluation
        if epoch % args.eva_epoch == 0:

            if netD.sparse_train_mode:
                d_current_keep_ratio, d_layer_keep_ratio = print_layer_keep_ratio(netD)
                print('D keep ratio: %.4f' % d_current_keep_ratio)
                with open('%s/D_keep_ratio_lambda_%s.txt' % (current_model_result_dir, str(args.lambda_)), 'a') as f:
                    f.write('epoch: %d' % epoch + '\n')
                    for x in d_layer_keep_ratio:
                        f.write(x + '\n')
                    f.write('Overall:' + str(d_current_keep_ratio.item()) + '\n')
                    f.write('\n')
            if netG.sparse_train_mode:
                print('--------------')
                g_current_keep_ratio, g_layer_keep_ratio = print_layer_keep_ratio(netG)
                print('G keep ratio: %.4f' % g_current_keep_ratio)
                with open('%s/G_keep_ratio_lambda_%s.txt' % (current_model_result_dir, str(args.lambda_)), 'a') as f:
                    f.write('epoch: %d' % epoch + '\n')
                    for x in g_layer_keep_ratio:
                        f.write(x + '\n')
                    f.write('Overall:' + str(g_current_keep_ratio.item()) + '\n')
                    f.write('\n')

            backup_param = copy_params(netG)
            load_params(netG, netG_avg_param)
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

            fid_record.append(fid)

            load_params(netG, backup_param)

            # avg_netG = deepcopy(netG)
            # load_params(avg_netG, netG_avg_param)

            if len(fid_record) >= 5:
                print(fid_record[-5], fid_record[-4], fid_record[-3], fid_record[-2], fid_record[-1])
                average_fid = 0.1 * fid_record[-5] + 0.1 * fid_record[-4] + 0.2 * fid_record[-3] + \
                              0.2 * fid_record[-2] + 0.4 * fid_record[-1]
                print(average_fid)

                if average_fid >= fid:
                    netG.sparse_train_mode = True
                    netD.sparse_train_mode = True
                else:
                    netG.sparse_train_mode = False
                    netD.sparse_train_mode = False
                set_training_mode(netG, netG.sparse_train_mode)
                set_training_mode(netD, netD.sparse_train_mode)

            netG.train()

if __name__ == '__main__':
    model_name = 'RG-SNGAN'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=800)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--image_size', type=int, default=32)
    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='./dataset')
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--lambda_', type=float, default=1e-12)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--eva_size', type=int, default=10000)
    argparser.add_argument('--eva_epoch', type=int, default=5)
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
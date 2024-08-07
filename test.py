import torch
import torch.nn.functional as F
from tqdm import tqdm
from dice_loss import dice_coeff
from torch.utils.data import DataLoader
from utils.dataset import BasicDataset
import logging
import argparse
import sys
from VGUNet import *
import numpy as np
import pandas as pd
# from skimage.io import  imsave

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0003,
                        help='Learning rate', dest='lr')
    # parser.add_argument('-f', '--load', dest='load', type=str, default=False,
    #                     help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    return parser.parse_args()


def main():
    args =get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_img = './data/prediction/images/'
    dir_mask = './data/prediction/maps/'
    dir_csv = './data/prediction/test.csv'
    dir_checkpoint = './data/prediction/checkpoints/'

    net = VGUNet(in_ch=3, out_ch=1, bilinear=True, fix_grad=True)
    net.eval()
    mask_type = torch.float32 if net.out_ch== 1 else torch.long

    test_dataset= BasicDataset(dir_img, dir_mask, dir_csv, scale=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
    
    n_test = len(test_loader)  # the number of batch
    tot = 0
    tot_regress_loss_val=0
    orient_t = torch.tensor([])
    orient_p = torch.tensor([])

    with tqdm(total=n_test, desc='Testing round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            imgs, true_masks = batch['image'], batch['mask']

            target= batch['orientation']
            #print("target: ", target)

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            target = target.type(torch.FloatTensor).to(device)

            target = target.squeeze()
            #print("target.squeeze: ", target)

            orient_t = torch.cat((orient_t, target), dim=0)
            #print("orient_t: ",orient_t )

            with torch.no_grad():
                mask_pred, regress_pred = net(imgs)
                #print("regress_pred: ", regress_pred)

                regress_pred = regress_pred.squeeze()
                #print("regress_pred.squeeze: ", regress_pred)


                orient_p = torch.cat((orient_p, regress_pred.cpu()), dim=0)
                #print("orient_p: ",orient_p)

            if net.out_ch > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

                # regress_loss_val= F.mse_loss(regress_pred, target)
                # tot_regress_loss_val += regress_loss_val.item()
            pbar.update()

    errors = np.abs(orient_t - orient_p)
    #print("errors: ",errors )

    #correct_predictions = torch.sum(errors < 0.12).item() # |orient_t - orient_p|=tan(5)sin(45)=0.08 sin(45)
    tresheld= 0.2 * np.abs(orient_t)
    print("tresheld: ",tresheld )
    correct_predictions = torch.sum(errors < tresheld).item()
    print("\n correct_predictions: ", correct_predictions)
    total_predictions = orient_t.size(0)* 3
    print("\n total_predictions: ", total_predictions)
    accuracy = correct_predictions / total_predictions * 100

    print("\n Regression accuracy : ", accuracy )
    print("\n Dice similarity coefficient for test dataset:", tot / n_test)

    # os.makedirs('./data/prediction/image_pred/', exist_ok=True)
    # imsave('./data/prediction/image_pred/'+".jpg",rgbPic)

if __name__ == '__main__':
    main( )
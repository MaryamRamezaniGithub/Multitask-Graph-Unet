import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.out_ch== 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_regress_loss_val=0

    orient_t = torch.tensor([]).to(device)
    orient_p = torch.tensor([]).to(device)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']

            target = batch['orientation']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            target = target.type(torch.FloatTensor).to(device)
            target = target.squeeze()
            orient_t = torch.cat((orient_t, target), dim=0)

            with torch.no_grad():
                mask_pred, regress_pred = net(imgs)
            regress_pred = regress_pred.squeeze()
            orient_p = torch.cat((orient_p, regress_pred), dim=0)

            if net.out_ch > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

                regress_loss_val= F.mse_loss(regress_pred, target)
                tot_regress_loss_val += regress_loss_val.item()
            pbar.update()
    errors = np.abs((orient_t - orient_p).cpu())
    #print("errors: ",errors )
    tresheld= 0.2 * np.abs((orient_t).cpu())
    correct_predictions = torch.sum(errors < tresheld).item() # |orient_t - orient_p|=tan(5)sin(45)=0.08 sin(45)
    print("\n correct_predictions: ", correct_predictions)
    total_predictions = orient_t.size(0)* 2
    print("\n total_predictions: ", total_predictions)
    accuracy = correct_predictions / total_predictions
    acc_result=round(accuracy, 5)

    dice_result= round(tot / n_val, 5)

    # return tot / n_val, tot_regress_loss_val / n_val , accuracy
    return dice_result*100 , tot_regress_loss_val / n_val , acc_result * 100

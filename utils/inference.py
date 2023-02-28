import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve, auc, average_precision_score
import torchvision.transforms as standard_transforms

from IPython import embed

def preprocess_image(x, mean_std):
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)
    x = x.cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x

def fpr_at_95_tpr(roc_tuple):
    fpr, tpr, _ = roc_tuple
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):    
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x>=0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def iter_over(net, image_list, mask_list, args, get_evals=False):
    __evals = []

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    anomaly_score_list = []
    ood_gts_list = []

    # Iterate over all images
    data_len = len(image_list)
    it = range(data_len)
    if args.local_rank == 0:
        it = tqdm(it, desc="Evaluating")
    for i in it:
        mask = mask_list[i]
        image = image_list[i]
        image = image.astype('uint8')
        ood_gts = mask
        ood_gts_list.append(np.expand_dims(ood_gts, 0))
        with torch.no_grad():
            img_ratios = args.inference_scale
            image2 = preprocess_image(image, mean_std)
            step_mains = []
            anomaly_score_accu = torch.zeros((1, *image.shape[:2]), device="cuda")
            for r in img_ratios:
                image_in = torch.nn.functional.interpolate(
                    image2, 
                    scale_factor=r, 
                    align_corners=True,
                    mode='bilinear'
                )
                main_out, anomaly_score = net(image_in, output_anomaly_score=True)
                step_mains += [main_out.cpu().clone()]
                del main_out
                anomaly_score_accu += torch.nn.functional.interpolate(
                    anomaly_score.unsqueeze(1),
                    size=anomaly_score_accu.shape[-2:],
                    align_corners=True,
                    mode="bilinear"
                ).squeeze(1)
                del anomaly_score
            anomaly_score_accu /= len(img_ratios)
            if get_evals:
                __evals += [step_mains]

        anomaly_score_list.append(anomaly_score_accu.cpu().numpy())

    # anomaly score, anomaly annotaion, segmentation logits
    return anomaly_score_list, ood_gts_list, __evals  

def metrics(anomaly_score_list, ood_gts_list):

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = -1 * anomaly_scores[ood_mask]
    ind_out = -1 * anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print('Calculating metrics...')

    fpr, tpr, ths = roc_curve(val_label, val_out)

    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(val_label, val_out)
    fpr_tpr95 = fpr_at_95_tpr((fpr, tpr, ths))

    return roc_auc, prc_auc, fpr_tpr95
    
from itertools import islice
import os
import torch
import datasets
import numpy as np
from tqdm import tqdm
from utils.misc import print0

def load_stats(net, ident):
    mean = np.load(f"stats/{ident}_mean.npy", allow_pickle=True).item()
    var = np.load(f"stats/{ident}_var.npy", allow_pickle=True).item()
    net.module.set_statistics(mean, var)
    
def calculate_statistics(net, ident, args):
    mean_file, var_file = f"stats/{ident}_mean.npy", f"stats/{ident}_var.npy"
    print0(f"Stat files {mean_file}, {var_file}")

    # skip existing mean and variance
    if os.path.exists(mean_file) and os.path.exists(var_file):
        print0("Stats exists")
        return

    args.ngpu = torch.distributed.get_world_size()
    args.bs_mult = 8
    sample_batch = int(120 / args.ngpu)
    train_loader, _, _, _ = datasets.setup_loaders(args)
    data_iter = islice(train_loader, sample_batch)
    if torch.distributed.get_rank() == 0:
        data_iter = tqdm(data_iter, total=sample_batch)

    net.eval()
    print0("Calculating statistics...")

    
    _zeros = lambda: torch.zeros(19).to(torch.float64).cuda()
    sum_x = _zeros()
    sum_x2 = _zeros()
    count = _zeros()
    sample_batch_count = torch.zeros(1).to(torch.float64).cuda()

    for data in data_iter:
        inputs = data[0]
        inputs = inputs.cuda()
        sample_batch_count += 1

        with torch.no_grad():
            outputs, _ = net(inputs) 
            del _
            pred_list = outputs.transpose(1, 3)
            pred_list, prediction = pred_list.max(3)
        for c in range(19):
            max_mask = pred_list[prediction == c]
            count[c] += len(max_mask)
            sum_x[c] += max_mask.sum().item()
            sum_x2[c] += (max_mask ** 2).sum().item()
        del outputs

    torch.distributed.all_reduce(sum_x, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(sum_x2, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(count, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(sample_batch_count, torch.distributed.ReduceOp.SUM)

    if torch.distributed.get_rank() != 0:
        return

    mean = sum_x / count
    var = sum_x2 / count - mean ** 2

    mean_dict, var_dict = {}, {}
    for c in range(datasets.num_classes):
        mean_dict[c] = mean[c].item()
        var_dict[c] = var[c].item()

    print0(f"class mean: {mean_dict}")
    print0(f"class var: {var_dict}")
    print0(f"pixel count: {count}")
    print0(f"batch count: {sample_batch_count}")

    np.save(mean_file, mean_dict)
    np.save(var_file, var_dict)

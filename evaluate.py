import time
from pathlib import Path

import numpy as np
from attrdict import AttrDict
import torch
from utils.calculate_statistics import calculate_statistics, load_stats
from anomaly_dataset import get_anomaly_dataset
from utils.inference import iter_over, metrics
from options import get_parser, init_cuda
import optimizer
import network
import shlex
import sys

if __name__ == "__main__":
    parser = get_parser()
    tmp_args = parser.parse_args()
    init_cuda(tmp_args)

    ckpt = torch.load(tmp_args.snapshot, map_location='cpu')
    cmd = ckpt['command']
    ckpt_args, other_args = get_parser().parse_known_args(shlex.split(cmd) + sys.argv[1:])
    ckpt_args.local_rank = tmp_args.local_rank

    net = network.get_net(ckpt_args, None, None)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, tmp_args.local_rank)
    epoch = optimizer.load_weights(net, None, None, None, False, ckpt)
    ident = f"{ckpt_args.tag}_{epoch}"
    net.eval()

    # calculate class mean and variance
    calculate_statistics(net, ident, tmp_args)
    torch.distributed.barrier()
    load_stats(net, ident)

    # load anomaly dataset
    image_list_all, mask_list_all = get_anomaly_dataset(tmp_args.anomaly_dataset)
    assert len(mask_list_all) == len(mask_list_all)
    ds_len = len(image_list_all)

    # split into all ranks
    image_each_proc = len(mask_list_all) // torch.distributed.get_world_size()
    res = len(mask_list_all) % torch.distributed.get_world_size()
    if tmp_args.local_rank < res:
        image_each_proc += 1
        pos = slice(image_each_proc * tmp_args.local_rank, image_each_proc * (tmp_args.local_rank + 1))
    else:
        pos = slice(res + image_each_proc * tmp_args.local_rank, res + image_each_proc * (tmp_args.local_rank + 1))
    image_list = image_list_all[pos]
    mask_list = mask_list_all[pos]
    del image_list_all, mask_list_all

    # get anomaly scores
    as_list, ood_list, evals = iter_over(net, image_list, mask_list, tmp_args)
    tmp_file_name = f"rank{torch.distributed.get_rank()}_{time.time()}.npz"
    np.savez(tmp_file_name, as_list, ood_list)

    # gather from all ranks
    names = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(names, tmp_file_name)

    # calculate metrics
    if tmp_args.local_rank == 0:
        as_list_total, ood_list_total = [], []
        for name in names:
            eval_results = np.load(name)
            as_list_total.append(eval_results['arr_0'])
            ood_list_total.append(eval_results['arr_1'])
            Path(name).unlink(missing_ok=True)
        assert len(as_list_total) == torch.distributed.get_world_size()
        del image_list, mask_list, as_list, ood_list, evals

        as_list_total = [a for r in as_list_total for a in r]
        ood_list_total = [o for r in ood_list_total for o in r]
        roc_auc, prc_auc, fpr_tpr95 = metrics(as_list_total, ood_list_total)
        print("Checkpoint:", tmp_args.snapshot)
        print("Dataset:", tmp_args.anomaly_dataset)
        print(f'AUROC: {roc_auc}')
        print(f'AUPRC: {prc_auc}')
        print(f'FPR@TPR95: {fpr_tpr95}')

"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg
from IPython import embed


def get_optimizer(args, net):
    assert args.adam, "Only admaW supported"
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_ITER == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_ITER
        scale_value = args.rescale
        lambda1 = lambda iteration: \
             math.pow(1 - iteration / args.max_iter,
                      args.poly_exp) if iteration < rescale_thresh else scale_value * math.pow(
                          1 - (iteration - rescale_thresh) / (args.max_iter - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        def warmup(args):
            def schedule_f(iteration):
                remaining = args.max_iter - args.warmup_iter
                exc_iteration = iteration - args.warmup_iter
                if iteration <= args.warmup_iter:
                    return iteration / args.warmup_iter
                else:
                    return math.pow(1 - exc_iteration / remaining, args.poly_exp)
            return schedule_f
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup(args))
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))
    
    return optimizer, scheduler


def load_weights(net, optimizer, scheduler, snapshot_file, restore_optimizer_bool=False, checkpoint=None):
    """
    Load weights from snapshot file
    """

    net, optimizer, scheduler, epoch, cmd = restore_snapshot(net, optimizer, scheduler, snapshot_file,
            restore_optimizer_bool, checkpoint=checkpoint)
    return epoch


def restore_snapshot(net, optimizer, scheduler, snapshot, restore_optimizer_bool, checkpoint=None):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    if not checkpoint:
        logging.info("Loading weights from model %s", snapshot)
        checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
        logging.info("Checkpoint Load Compelete")
    else:
        logging.info("Use Given Checkpoint")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer, scheduler, checkpoint['epoch'], checkpoint["command"]


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            ...
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_copy(target_net, source_net):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = target_net.state_dict()
    loaded_dict = source_net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
            print("Matched", k)
        else:
            print("Skipped loading parameter ", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    target_net.load_state_dict(net_state_dict)
    return target_net

import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import numpy as np
import options
import random

# set_seed(cfg.RANDOM_SEED)
random_seed = cfg.RANDOM_SEED  #304
print("RANDOM_SEED", random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = options.get_parser()
args = parser.parse_args()
options.init_cuda(args)

def main():
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_loader, val_loaders, train_obj, _ = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, _ = optimizer.load_weights(net, optim, scheduler, args.snapshot, args.restore_optimizer)
        epoch += 1
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    while i < args.max_iter:
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
        train_loader.sampler.set_epoch(epoch + 1)
        if i % args.val_interval == 0:
            for dataset, val_loader in val_loaders.items():
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
        else:
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        epoch += 1

    # Validation after epochs

    assert len(val_loaders) == 1

    for dataset, val_loader in val_loaders.items():
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)


def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)
    print("current iteration is", curr_iter, " | current epoch is", curr_epoch)

    # # load clip model
    # embeds = get_label_matrix()

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break
        inputs, seg_gts, ood_gts, _, aux_gts = data

        B, C, H, W = inputs.shape
        num_domains = 1
        inputs = [inputs]
        seg_gts = [seg_gts]
        ood_gts = [ood_gts]
        aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, seg_gts, ood_gts, aux_gts)):
            input, seg_gt, ood_gt, aux_gt = ingredients

            start_ts = time.time()

            input, seg_gt, ood_gt = input.cuda(), seg_gt.cuda(), ood_gt.cuda()

            optim.zero_grad()
            outputs = net(input, seg_gts=seg_gt, aux_gts=aux_gt)
            main_loss, aux_loss, anomaly_score = outputs
            total_loss = main_loss + (0.4 * aux_loss)

            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss

            if args.local_rank == 0:
                if i % 20 == 19:
                    msg = '[epoch {}], [iter {} / {} : {}], [total loss {:0.6f}], [seg loss {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                        main_loss.item(), time_meter.avg / args.train_batch_size)

                    logging.info(msg)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    writer.add_scalar('loss/main_loss', (main_loss.item()),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    # embeds = get_label_matrix()
  
    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, seg_gts, ood_gts, img_names, _ = data

        assert len(inputs.size()) == 4 and len(seg_gts.size()) == 3
        assert inputs.size()[2:] == seg_gts.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs = inputs.cuda()
        seg_gts_cuda = seg_gts.cuda()

        with torch.no_grad():
            main_out, anomaly_score = net(inputs)

        del inputs

        assert main_out.size()[2:] == seg_gts.size()[1:]
        assert main_out.size()[1] == datasets.num_classes

        main_loss = criterion(main_out, seg_gts_cuda)
        val_loss.update(main_loss.item(), batch_pixel_size)
        del seg_gts_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = main_out.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        iou_acc += fast_hist(predictions.numpy().flatten(), seg_gts.numpy().flatten(),
                             datasets.num_classes)
        del main_out, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc,
                    writer, curr_epoch, save_pth=save_pth)

    return val_loss.avg

if __name__ == '__main__':
    main()


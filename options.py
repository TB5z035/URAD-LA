import argparse
from optparse import check_choice
from re import T
import torch
import os
import time

def get_parser():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                        help='Network architecture.')
    parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                        help='a list of datasets; cityscapes')
    parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                        help='uniformly sample images across the multiple source domains')
    parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes'],
                        help='validation dataset list')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--cv', type=int, default=0,
                        help='cross-validation split id to use. Default # of splits set to 3 in config')
    parser.add_argument('--class_uniform_pct', type=float, default=0,
                        help='What fraction of images is uniformly sampled')
    parser.add_argument('--class_uniform_tile', type=int, default=1024,
                        help='tile size for class uniform sampling')
    parser.add_argument('--coarse_boost_classes', type=str, default=None,
                        help='use coarse annotations to boost fine data with specific classes')

    parser.add_argument('--img_wt_loss', action='store_true', default=False,
                        help='per-image class-weighted loss')
    parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                        help='class-weighted loss')
    parser.add_argument('--batch_weighting', action='store_true', default=False,
                        help='Batch weighting for class (use nll class weighting using batch stats')

    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')
    parser.add_argument('--strict_bdr_cls', type=str, default='',
                        help='Enable boundary label relaxation for specific classes')
    parser.add_argument('--rlx_off_iter', type=int, default=-1,
                        help='Turn off border relaxation after specific epoch count')
    parser.add_argument('--rescale', type=float, default=1.0,
                        help='Warm Restarts new learning rate ratio compared to original lr')
    parser.add_argument('--repoly', type=float, default=1.5,
                        help='Warm Restart new poly exp')

    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use Nvidia Apex AMP')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='parameter used by apex library')

    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--freeze_trunk', action='store_true', default=False)

    parser.add_argument('--hardnm', default=0, type=int,
                        help='0 means no aug, 1 means hard negative mining iter 1,' +
                        '2 means hard negative mining iter 2')

    parser.add_argument('--trunk', type=str, default='resnet101',
                        help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('--max_epoch', type=int, default=180)
    parser.add_argument('--max_iter', type=int, default=30000)
    parser.add_argument('--max_cu_epoch', type=int, default=100000,
                        help='Class Uniform Max Epochs')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--rrotate', type=int,
                        default=0, help='degree of random roate')
    parser.add_argument('--color_aug', type=float,
                        default=0.0, help='level of color augmentation')
    parser.add_argument('--gblur', action='store_true', default=False,
                        help='Use Guassian Blur Augmentation')
    parser.add_argument('--bblur', action='store_true', default=False,
                        help='Use Bilateral Blur Augmentation')
    parser.add_argument('--lr_schedule', type=str, default='poly',
                        help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='polynomial LR exponent')
    parser.add_argument('--bs_mult', type=int, default=2,
                        help='Batch size for training per gpu')
    parser.add_argument('--bs_mult_val', type=int, default=1,
                        help='Batch size for Validation per gpu')
    parser.add_argument('--crop_size', type=int, default=720,
                        help='training crop size')
    parser.add_argument('--pre_size', type=int, default=None,
                        help='resize image shorter edge to this before augmentation')
    parser.add_argument('--scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--restore_optimizer', action='store_true', default=False)

    parser.add_argument('--city_mode', type=str, default='train',
                        help='experiment directory date name')
    parser.add_argument('--workdir', type=str, default='workdir')
    parser.add_argument('--syncbn', action='store_true', default=True,
                        help='Use Synchronized BN')
    parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                        help='Dump Augmentated Images for sanity check')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Minimum testing to verify nothing failed, ' +
                        'Runs code for 1 epoch of train and val')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                        help='Weight Scaling for the losses')
    parser.add_argument('--maxSkip', type=int, default=0,
                        help='Skip x number of  frames of video augmented dataset')
    parser.add_argument('--scf', action='store_true', default=False,
                        help='scale correction factor')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--backbone_lr', type=float, default=-1.0,
                        help='different learning rate on backbone network')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='pooling methods, average is better than max')
    parser.add_argument('--warmup_iter', type=int, default=-1)

    parser.add_argument('--score_mode', type=str, default='bsl',
                        choices=['bsl', 'sml', 'ml', 'msp', 'entropy'])

    # Boundary suppression configs
    parser.add_argument('--enable_boundary_suppression', type=lambda x: x.lower() == 'true', default=False, help='enable boundary suppression')
    parser.add_argument('--boundary_width', type=int, default=4, help='initial boundary suppression width')
    parser.add_argument('--boundary_iteration', type=int, default=4, help='the number of boundary iterations')

    # Dilated smoothing configs
    parser.add_argument('--enable_dilated_smoothing', type=lambda x: x.lower() == 'true', default=True, help='enable dilated smoothing')
    parser.add_argument('--smoothing_kernel_size', type=int, default=7, help='kernel size of dilated smoothing')
    parser.add_argument('--smoothing_kernel_dilation', type=int, default=4, help='kernel dilation rate of dilated smoothing')

    parser.add_argument('--disable_le', default=False, action="store_true")
    parser.add_argument('--logit_type', default='binary', type=str)
    parser.add_argument('--temp', type=str, default='fixed')
    parser.add_argument('--T', type=float, default=0.07)
    parser.add_argument('--tau', default=0.8, type=float)
    parser.add_argument('--inf_temp', type=float, default=1.0)
    parser.add_argument('--inference_scale', type=float, nargs='+', default=[1.0])
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--anomaly_dataset', type=str, choices=['fslaf', 'laf', 'ra'])
    return parser

def init_cuda(args):
    if 'WORLD_SIZE' in os.environ:
        # args.apex = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
        print("Total world size: ", int(os.environ['WORLD_SIZE']))

    # args.world_size = 3
    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)
    # Initialize distributed communication
    args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

    torch.distributed.init_process_group(backend='nccl',
                                        init_method=args.dist_url,
                                        world_size=args.world_size,
                                        rank=args.local_rank)
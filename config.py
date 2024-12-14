"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import munch
import importlib
import os
from os.path import join as ospj
import shutil
import warnings

from util import Logger

_DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages')
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50', 'inception_v3')
_METHOD_NAMES = ('cam', 'adl', 'acol', 'spg', 'has', 'cutmix', 'crop', 'da', 'brid')
_SPLITS = ('train', 'val', 'test')

# For add normalized method
_NORM_NAMES = ('minmax', 'max', 'pas', 'ivr')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def box_v2_metric(args):
    if args.box_v2_metric:
        args.multi_contour_eval = True
        args.multi_iou_eval = True
    else:
        args.multi_contour_eval = False
        args.multi_iou_eval = False
        warnings.warn("MaxBoxAcc metric is deprecated.")
        warnings.warn("Use MaxBoxAccV2 by setting args.box_v2_metric to True.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_architecture_type(wsol_method):
    if wsol_method in ('has', 'cutmix', 'crop', 'da'):
        architecture_type = 'cam'
    else:
        architecture_type = wsol_method
    return architecture_type


def configure_data_paths(args):
    train = val = test = ospj(args.data_root, args.dataset_name)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def configure_mask_root(args):
    mask_root = ospj(args.mask_root, 'OpenImages')
    return mask_root


def configure_scoremap_output_paths(args):
    scoremaps_root = ospj(args.log_folder, 'scoremaps')
    scoremaps = mch()
    for split in ('train', 'val', 'test'):
        scoremaps[split] = ospj(scoremaps_root, split)
        if not os.path.isdir(scoremaps[split]):
            os.makedirs(scoremaps[split])
    return scoremaps


def configure_log_folder(args):
    log_folder = ospj('train_log', args.experiment_name)

    if os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
        else:
            raise RuntimeError("Experiment with the same name exists: {}"
                               .format(log_folder))
    os.makedirs(log_folder)
    return log_folder


def configure_log(args):
    log_file_name = ospj(args.log_folder, 'log.log')
    Logger(log_file_name)


def configure_reporter(args):
    reporter = importlib.import_module('util').Reporter
    reporter_log_root = ospj(args.log_folder, 'reports')
    if not os.path.isdir(reporter_log_root):
        os.makedirs(reporter_log_root)
    return reporter, reporter_log_root


def configure_pretrained_path(args):
    pretrained_path = None
    return pretrained_path


def check_dependency(args):
    if args.dataset_name == 'CUB':
        if args.num_val_sample_per_class >= 6:
            raise ValueError("num-val-sample must be <= 5 for CUB.")
    if args.dataset_name == 'OpenImages':
        if args.num_val_sample_per_class >= 26:
            raise ValueError("num-val-sample must be <= 25 for OpenImages.")


def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--override_cache', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')

    # Data
    parser.add_argument('--dataset_name', type=str, default='CUB',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                        default='dataset/',
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')

    # Setting
    parser.add_argument('--architecture', default='resnet18',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: resnet18)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre_trained model.')
    parser.add_argument('--cam_curve_interval', type=float, default=.001,
                        help='CAM curve interval')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--eval_checkpoint_type', type=str, default='last',
                        choices=('best', 'last'))
    parser.add_argument('--box_v2_metric', type=str2bool, nargs='?',
                        const=True, default=True)

    # Common hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_decay_frequency', type=int, default=30,
                        help='How frequently do we decay the learning rate?')
    parser.add_argument('--lr_classifier_ratio', type=float, default=10,
                        help='Multiplicative factor on the classifier layer.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)

    # Method-specific hyperparameters
    parser.add_argument('--wsol_method', type=str, default='base',
                        choices=_METHOD_NAMES)
    parser.add_argument('--has_grid_size', type=int, default=4)
    parser.add_argument('--has_drop_rate', type=float, default=0.5)
    parser.add_argument('--acol_threshold', type=float, default=0.7)
    parser.add_argument('--spg_threshold_1h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=0.01,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=0.5,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=0.05,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=0.1,
                        help='SPG threshold')
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--cutmix_beta', type=float, default=1.0,
                        help='CutMix beta')
    parser.add_argument('--cutmix_prob', type=float, default=1.0,
                        help='CutMix Mixing Probability')
    # Load Checkpoint used path
    parser.add_argument('--checkpoint_path', type=str, default=None)

    # Evaluated by SizeAvg
    parser.add_argument('--eval_size_ratio', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--box_size_threshold_list', nargs='+', type=int, default=[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

    # Evaluated by MaxBoxAccV2
    parser.add_argument('--eval_only_localization', type=str2bool, nargs='?',
                        const=True, default=False)

    # evaluate box size by answer box or max box
    parser.add_argument('--eval_max_size', type=str2bool, nargs='?',
                        const=True, default=False)

    # Change resolution of CAM
    parser.add_argument('--original_feature_map', type=str2bool, nargs='?', 
                        const=True, default=False)
    # For Crop Method hyperparameter
    parser.add_argument('--crop_threshold', type=float, default=0.15,
                        help='Crop cam threshold')
    parser.add_argument('--loss_ratio', type=float, default=0.5,
                        help='For Both Loss Margin')
    parser.add_argument('--loss_pos', type=float, default=0.5,
                        help='For Attention Loss Margin')
    parser.add_argument('--loss_neg', type=float, default=0.5,
                        help='For Attention Loss Margin')
    parser.add_argument('--crop_ratio', type=float, default=0.1,
                        help='For Choose Crop CAM Ratio')

    # Evaluate by val or test
    parser.add_argument('--eval_on_val_and_test', type=str2bool, nargs='?',
                        const=True, default=True)
    # Compare CAM size and real ground truth box size
    parser.add_argument('--compare_box_size', type=str2bool, nargs='?',
                        const=True, default=False)
    # paste after interpolation
    parser.add_argument('--large_cam', type=str2bool, nargs='?',
                        const=True, default=False)
    # For Training only Classifier
    parser.add_argument('--train_only_classifier', type=str2bool, nargs='?',
                        const=True, default=False)
    # For Class Agonistic Activation Map
    parser.add_argument('--attention_cam', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--use_inception_c', type=str2bool, nargs='?',
                        const=True, default=False)
    # Crop Start Time (For CUB)
    parser.add_argument('--crop_start_epoch', type=int, default=0,
                        help='Crop Start Epoch')

    # For Fuse with other method.
    parser.add_argument('--crop', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--other_method_loss_ratio', type=float, default=1)
    parser.add_argument('--crop_method_loss_ratio', type=float, default=1)

    # For Top1-Loc
    parser.add_argument('--top1_loc', type=str2bool, nargs='?',
                        const=True, default=False)

    # Method Training with Normalization method.
    parser.add_argument('--crop_with_norm', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--norm_method', type=str, default='minmax',
                        choices=_NORM_NAMES)
    parser.add_argument('--percentile', type=float, default=0.6)

    # For Brid
    parser.add_argument('--drop_threshold', type=float, default=0.7)
    parser.add_argument('--drop_prob', type=float, default=1)

    args = parser.parse_args()

    check_dependency(args)
    args.log_folder = configure_log_folder(args)
    configure_log(args)
    box_v2_metric(args)

    args.architecture_type = get_architecture_type(args.wsol_method)
    args.data_paths = configure_data_paths(args)
    args.metadata_root = ospj(args.metadata_root, args.dataset_name)
    args.mask_root = configure_mask_root(args)
    args.scoremap_paths = configure_scoremap_output_paths(args)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = configure_pretrained_path(args)
    args.spg_thresholds = ((args.spg_threshold_1h, args.spg_threshold_1l),
                           (args.spg_threshold_2h, args.spg_threshold_2l),
                           (args.spg_threshold_3h, args.spg_threshold_3l))

    return args

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

import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
from collections import OrderedDict

from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer
from util import string_contains_any
import wsol
import wsol.method

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': [],
        'resnet': ['conv1', 'layer1.', 'layer2.', 'layer3.', 'layer4.', 'fc.',
                   'toplayer.', 'smooth1.', 'smooth2.', 'smooth3.', 'smooth4.', 'smooth5.',
                   'latlayer1.', 'latlayer2.','latlayer3.','latlayer4.','latlayer5.'],
        'inception': [],
    }

    def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print(self.args)
        self._set_feature_param_layer_patterns()
        if self.args.wsol_method == 'crop' or self.args.crop:
            self._EVAL_METRICS = ['loss', 'att_loss', 'cls_loss', 'classification', 'localization']
        else:
            self._EVAL_METRICS = ['loss', 'classification', 'localization']
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = self._set_optimizer()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)
        # For Method of Module
        self.crop_module = wsol.method.CropCAM(self.args.large_feature_map,
                                               self.args.original_feature_map,
                                               architecture=self.args.architecture,
                                               # Hyperparameters
                                               loss_ratio=self.args.loss_ratio,
                                               loss_pos=self.args.loss_pos,
                                               loss_neg=self.args.loss_neg,
                                               crop_threshold=self.args.crop_threshold,
                                               crop_ratio=self.args.crop_ratio,
                                               # For CAAM
                                               attention_cam=self.args.attention_cam,
                                               # For attach the network freely.
                                               wsol_method=self.args.wsol_method,
                                               other_method_loss_ratio=self.args.other_method_loss_ratio,
                                               crop_method_loss_ratio=self.args.crop_method_loss_ratio,
                                               # For Several Norm Method.
                                               norm_method=self.args.norm_method,
                                               percentile=self.args.percentile,
                                               crop_with_norm=self.args.crop_with_norm)

        self.da_module = wsol.method.DACAM(num_classes=self._NUM_CLASSES_MAPPING[self.args.dataset_name],
                                           architecture=self.args.architecture,
                                           dataset_name=self.args.dataset_name,
                                           model=self.model)

    def _set_feature_param_layer_patterns(self):
        if self.args.train_only_classifier:
            self._FEATURE_PARAM_LAYER_PATTERNS = {
                'vgg': ['features.'],
                'resnet': ['layer4.', 'last_conv.', 'fc.'],
                'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                              'Conv2d_3', 'Conv2d_4'],
            }

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric in ('loss', 'cls_loss', 'att_loss') else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,
            original_feature_map=self.args.original_feature_map,
            attention_cam=self.args.attention_cam,
            use_inception_c=self.args.use_inception_c,
            drop_threshold=self.args.drop_threshold,
            drop_prob=self.args.drop_prob,
        )
        # To Use Every GPU
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.cuda()
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_features.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_classifiers.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _wsol_training(self, images, target, epoch=None):
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsol.method.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.cross_entropy_loss(logits, target_a) * lam +
                    self.cross_entropy_loss(logits, target_b) * (1. - lam))
            return logits, loss

        if self.args.wsol_method == 'has':
            images = wsol.method.has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)
        # For Crop Methods
        if self.args.wsol_method == 'crop' or self.args.crop:
            if epoch >= self.args.crop_start_epoch:
                output_dict = self.crop_module.forward(self.model, images, target)
                logits = output_dict["logits"]
                loss, att_loss, cls_loss = self.crop_module.get_loss(output_dict=output_dict, target=target)
                return logits, loss, att_loss, cls_loss

        # For DA Methods
        if self.args.wsol_method == 'da':
            if self.args.crop and epoch >= self.args.crop_start_epoch:
                output_dict = self.crop_module.forward(self.da_module, images, target)
                logits = output_dict["logits"]
                loss, att_loss, cls_loss = self.crop_module.get_loss(output_dict=output_dict, target=target)
            else:
                att_loss = torch.tensor(0)
                cls_loss = torch.tensor(0)
                logits, loss = self.da_module.forward(self.model, images, target)
            return logits, loss, att_loss, cls_loss

        output_dict = self.model(images, target)
        logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
                output_dict=output_dict,
                target=target,
                spg_thresholds=self.args.spg_thresholds)
        else:
            loss = self.cross_entropy_loss(logits, target)

        if self.args.wsol_method == 'crop':
            return logits, loss, torch.tensor(0), loss

        return logits, loss

    def train(self, split, epoch):
        self.model.train()
        loader = self.loaders[split]

        total_loss = 0.0
        total_att_loss = 0.0
        total_cls_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, images_id) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            if self.args.wsol_method == 'crop' or self.args.crop:
                logits, loss, att_loss, cls_loss = self._wsol_training(images, target, epoch)
                total_att_loss += att_loss.item() * images.size(0)
                total_cls_loss += cls_loss.item() * images.size(0)
            else:
                logits, loss = self._wsol_training(images, target, epoch)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            if not torch.isnan(loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        loss_average = total_loss / float(num_images)
        if self.args.wsol_method == 'crop' or self.args.crop:
            att_loss_average = total_att_loss / float(num_images)
            cls_loss_average = total_cls_loss / float(num_images)
            self.performance_meters[split]['att_loss'].update(att_loss_average)
            self.performance_meters[split]['cls_loss'].update(cls_loss_average)

        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            output_dict = self.model(images)
            pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, epoch, split):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
            box_size_threshold_list=self.args.box_size_threshold_list,
            eval_size_ratio=self.args.eval_size_ratio,
            eval_max_size=self.args.eval_max_size,
            norm_method=self.args.norm_method,
            percentile=self.args.percentile,
            crop_module=self.crop_module,
            wsol_method=self.args.wsol_method,
            compare_box_size=self.args.compare_box_size,
            data_root=self.args.data_paths,
            epoch=epoch,
            da_module=self.da_module,
            crop=self.args.crop,
            top1_loc=self.args.top1_loc,
        )
        cam_performance = cam_computer.compute_and_evaluate_cams()

        if self.args.eval_size_ratio:
            for box_size_threshold in self.args.box_size_threshold_list[:-1]:
                if self.args.multi_contour_eval or self.args.dataset_name == 'OpenImages':
                    cam_performance[box_size_threshold] = np.average(cam_performance[box_size_threshold])
                else:
                    cam_performance[box_size_threshold] = cam_performance[box_size_threshold][
                        self.args.iou_threshold_list.index(50)]

            for _SIZE_PREV, _SIZE_AFTER in zip(self.args.box_size_threshold_list[:-1], self.args.box_size_threshold_list[1:]):
                print('box size ratio: {}~{}, localization : {}'.format(_SIZE_PREV, _SIZE_AFTER, cam_performance[_SIZE_PREV]))

        else:
            if self.args.top1_loc or self.args.dataset_name != 'OpenImages':
                loc_score = cam_performance[self.args.iou_threshold_list.index(50)]
            elif self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
                loc_score = np.average(cam_performance)
            else:
                loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

            self.performance_meters[split]['localization'].update(loc_score)
            print('Final Localization Score :', loc_score)

            if self.args.dataset_name in ('CUB', 'ILSVRC'):
                for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                    self.performance_meters[split][
                        'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                        cam_performance[idx])

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(

                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))

    def load_checkpoint_used_path(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            new_state_dict = OrderedDict()
            checkpoint = torch.load(checkpoint_path)
            # For DataParallel
            if torch.cuda.device_count() == 1:
                for k, v in checkpoint['state_dict'].items():
                    if "module." in k:
                        name = k[7:]
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                self.model.load_state_dict(new_state_dict)
            else:
                for k, v in checkpoint['state_dict'].items():
                    if "module." not in k:
                        name = "module." + k
                        new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    trainer = Trainer()

    if trainer.args.eval_size_ratio or trainer.args.eval_only_localization:
        print("===========================================================")
        print("Start evaluation on test set ...")
        trainer.load_checkpoint_used_path(checkpoint_path=trainer.args.checkpoint_path)
        if trainer.args.eval_on_val_and_test:
            trainer.evaluate(trainer.args.epochs, split='val')
        else:
            trainer.evaluate(trainer.args.epochs, split='test')
    else:
        print("===========================================================")
        print("Start epoch 0 ...")
        if trainer.args.checkpoint_path is not None:
            trainer.load_checkpoint_used_path(checkpoint_path=trainer.args.checkpoint_path)
        trainer.evaluate(epoch=0, split='val')
        trainer.print_performances()
        trainer.report(epoch=0, split='val')
        trainer.save_checkpoint(epoch=0, split='val')
        print("Epoch 0 done.")

        for epoch in range(trainer.args.epochs):
            print("===========================================================")
            print("Start epoch {} ...".format(epoch + 1))
            trainer.adjust_learning_rate(epoch + 1)
            train_performance = trainer.train(split='train', epoch=epoch)
            trainer.report_train(train_performance, epoch + 1, split='train')
            trainer.evaluate(epoch + 1, split='val')
            trainer.print_performances()
            trainer.report(epoch + 1, split='val')
            trainer.save_checkpoint(epoch + 1, split='val')
            print("Epoch {} done.".format(epoch + 1))

        print("===========================================================")
        print("Final epoch evaluation on test set ...")

        trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
        trainer.evaluate(trainer.args.epochs, split='test')
        trainer.print_performances()
        trainer.report(trainer.args.epochs, split='test')
        trainer.save_performances()


if __name__ == '__main__':
    main()

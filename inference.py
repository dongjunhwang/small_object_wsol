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

import cv2
import numpy as np
import os
import random
from os.path import join as ospj
from os.path import dirname as ospd

from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from util import t2n
from wsol.method import crop

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root, iou_threshold_list, dataset_name, split,
                 multi_contour_eval, box_size_threshold_list, eval_size_ratio, eval_max_size,
                 crop_module, wsol_method, compare_box_size, data_root, epoch, da_module, crop, top1_loc,
                 norm_method='minmax', percentile=0.6, cam_curve_interval=.001, log_folder=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          cam_curve_interval=cam_curve_interval,
                                          multi_contour_eval=multi_contour_eval,
                                          box_size_threshold_list=box_size_threshold_list,
                                          eval_size_ratio=eval_size_ratio,
                                          eval_max_size=eval_max_size,
                                          compare_box_size=compare_box_size,
                                          top1_loc=top1_loc,)
        # To use GradCAM

        # To use variant normalization
        self.norm_method = norm_method
        self.percentile = percentile

        # To use Crop Method
        self.crop_module = crop_module
        self.wsol_method = wsol_method

        # To use DA Method
        self.da_module = da_module
        self.crop = crop

        # For Make Qualitative CAM
        self.data_root = data_root[split]
        self.dataset_name = dataset_name
        self.count_qualitative_cam = 0
        self.grid_size = (20, 20)
        self.list_qualitative_cam = [None] * self.grid_size[0]
        self.random_weight = {"ILSVRC": [1, 9], "CUB": [1, 0], "OpenImages": [1, 0]}
        self.epoch = epoch

        # For Top1 Loc
        self.top1_loc = top1_loc

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        scoremap_expectation = [0, 0, 0]
        scoremap_variance = [0, 0, 0]
        image_len = 0
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            targets = targets.cuda()
            if self.wsol_method == "crop" or self.crop:
                cams, logits = self.crop_module.forward(self.model, images,
                                                        targets, return_cam=True)
                cams = t2n(cams)
            elif self.wsol_method == "da":
                if self.crop:
                    cams, logits = self.crop_module.forward(self.da_module, images,
                                                            targets, return_cam=True)
                    cams = t2n(cams)
                else:
                    cams, logits = self.da_module(images, targets, return_cam=True)
                    cams = t2n(cams)
            elif self.wsol_method == "brid":
                output_dict = self.model(images, targets, return_cam=True)
                cams, logits = t2n(output_dict['cams']), output_dict['logits']
            else:
                cams, logits = self.model(images, targets, return_cam=True)
                cams = t2n(cams)

            for cam, image_id, target, logit in zip(cams, image_ids, targets, logits):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = self.normalize_scoremap(cam_resized)
                if self.count_qualitative_cam <= self.grid_size[0] * self.grid_size[1] \
                        and random.choices([True, False], weights=self.random_weight[self.dataset_name])[0]:
                    self.qualitative_grid_cam(cam_normalized, image_id)
                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                image_len += 1
                top1_cls = True if target == logit.argmax(dim=0) else False
                self.evaluator.accumulate(cam_normalized, image_id, top1_cls)

        return self.evaluator.compute()

    def make_cam(self, input_img, scoremap):
        heatmap_resize = cv2.resize(scoremap, (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST)
        heatmap = np.uint8(255 * heatmap_resize.squeeze())
        heatmap_native = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heatmap = heatmap_native * 0.5 + input_img
        heatmap = heatmap

        return heatmap.squeeze(), heatmap_native.squeeze()

    def qualitative_grid_cam(self, cam, image_id):
        # For Make Qualitative CAM
        input_img = cv2.imread(ospj(self.data_root, image_id))
        input_img = cv2.resize(input_img, (cam.shape[0], cam.shape[1]))
        heatmap, _ = self.make_cam(input_img, cam)
        if self.dataset_name != "OpenImages":
            x_resize = cam.shape[0] / self.evaluator.image_sizes[image_id][0]
            y_resize = cam.shape[1] / self.evaluator.image_sizes[image_id][1]
            for box in self.evaluator.original_bboxes[image_id]:
                x1, y1, x2, y2 = map(int, box)
                x1 *= x_resize
                x2 *= x_resize
                y1 *= y_resize
                y2 *= y_resize
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                heatmap = cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 0, 255), 2)
        heatmap = heatmap.astype(int)
        if self.count_qualitative_cam < self.grid_size[0] * self.grid_size[1]:
            ri = int(self.count_qualitative_cam/self.grid_size[0])
            if self.list_qualitative_cam[ri] is None:
                self.list_qualitative_cam[ri] = [heatmap]
            else:
                self.list_qualitative_cam[ri].append(heatmap)
            self.count_qualitative_cam += 1
        elif self.count_qualitative_cam == self.grid_size[0] * self.grid_size[1] and self.epoch % 5 == 0:
            for i in range(self.grid_size[0]):
                self.list_qualitative_cam[i] = np.hstack(self.list_qualitative_cam[i])
            qualitative_cam = np.vstack(self.list_qualitative_cam)
            qualitative_cam_path = ospj(self.log_folder, f'qualitative_cam_{self.epoch}.jpg')
            if not os.path.exists(ospd(self.log_folder)):
                os.makedirs(ospd(self.log_folder))
            cv2.imwrite(ospj(qualitative_cam_path), qualitative_cam)
            self.count_qualitative_cam += 1

    def normalize_scoremap(self, cam):
        """
        Args:
            cam: numpy.ndarray(size=(H, W), dtype=np.float)
        Returns:
            numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
            If input array is constant, a zero-array is returned.
        """
        if np.isnan(cam).any():
            return np.zeros_like(cam)
        if cam.min() == cam.max():
            return np.zeros_like(cam)
        if self.norm_method == 'minmax':
            cam -= cam.min()
            cam /= cam.max()
        elif self.norm_method == 'max':
            cam = np.maximum(0, cam)
            cam /= cam.max()
        elif self.norm_method == 'pas':
            cam -= cam.min()
            cam_copy = cam.flatten()
            cam_copy.sort()
            maxx = cam_copy[int(cam_copy.size * 0.9)]
            cam /= maxx
            cam = np.minimum(1, cam)
        elif self.norm_method == 'ivr':
            cam_copy = cam.flatten()
            cam_copy.sort()
            minn = cam_copy[int(cam_copy.size * self.percentile)]
            cam -= minn
            cam = np.maximum(0, cam)
            cam /= cam.max()
        else:
            print('Norm not defined')
        return cam

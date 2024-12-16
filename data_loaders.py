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

import munch
import numpy as np
import os
import torch
import random

from PIL import Image
from util import check_box_convention
from imutil import pil_resize
from imutil import pil_rescale
from imutil import TorchvisionNormalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = ('train', 'val', 'test', 'small')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def square_crop_function(crop_range, max_w, max_h):
    x, y, w, h = map(int, crop_range)
    if w < h:
        expand_x = int(x - (h - w) / 2)
        x = expand_x if expand_x > 0 else 0
        w = h if x + h < max_w else max_w - x
    elif w > h:
        expand_y = int(y - (w - h) / 2)
        y = expand_y if expand_y > 0 else 0
        h = w if y + w < max_h else max_h - y

    return x, y, w, h


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def configure_metadata(metadata_root, openimages_box,
                       point_loc=False, proxy_point=False, box_loc=False,
                       with_small=False, small_count=100):
    metadata = mch()

    if with_small:
        file_path = os.path.join(metadata_root, f'image_ids_with_small_{small_count}.txt')
        if os.path.exists(file_path):
            metadata.image_ids = file_path
        else:
            metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    else:
        metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')

    metadata.image_ids_proxy = os.path.join(metadata_root,
                                            'image_ids_proxy.txt')

    if with_small:
        file_path = os.path.join(metadata_root, 'class_labels_with_small.txt')
        if os.path.exists(file_path):
            metadata.class_labels = file_path
        else:
            metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    else:
        metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    if openimages_box or box_loc:
        metadata.localization = os.path.join(metadata_root, 'localization_box.txt')
        metadata.localization_mask = os.path.join(metadata_root, 'localization.txt')
    else:
        metadata.localization = os.path.join(metadata_root, 'localization.txt')

    if point_loc:
        metadata.point_loc = os.path.join(metadata_root, 'point_loc.txt')

    if proxy_point:
        metadata.image_ids_proxy_point = os.path.join(metadata_root,
                                                      'image_ids_proxy_point.txt')

    return metadata


def get_image_ids(metadata, proxy=False, proxy_point=False):
    """
    image_ids_with_small_100.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    suffix = '_proxy_point' if proxy_point else suffix
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata, multi_label=False):
    """
    image_ids_with_small_100.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            if multi_label:
                line_list = line.strip('\n').split(',')
                class_labels[line_list[0]] = [i for i in line_list[1:]]
            else:
                image_id, class_label_string = line.strip('\n').split(',')
                class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes


def get_mask_paths(metadata, box_loc=False, multi_label=False):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    metadata_path = metadata.localization_mask if box_loc else metadata.localization
    metadata_path = metadata.class_labels if multi_label else metadata_path
    with open(metadata_path) as f:
        for line in f.readlines():
            if multi_label:
                line_list = line.strip('\n').split(',')
                image_id = line_list[0]
                mask_paths[image_id] = {}
                for i in line_list[1:]:
                    mask_paths[image_id][int(i)] = image_id.split('.jpg')[0]+"_"+str(i)+".png"
            else:
                image_id, mask_path, ignore_path = line.strip('\n').split(',')
                if image_id in mask_paths:
                    mask_paths[image_id].append(mask_path)
                    assert (len(ignore_path) == 0)
                else:
                    mask_paths[image_id] = [mask_path]
                    ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def get_point_loc(metadata):
    """
    point_loc.txt has the structure

    <path>,<x>,<y>
    path/to/image1.jpg,215,162
    path/to/image2.jpg,239,211
    path/to/image3.jpg,137,201
    ...
    """
    image_point = {}
    with open(metadata.point_loc) as f:
        for line in f.readlines():
            image_id, x, y = line.strip('\n').split(',')
            x, y = int(x), int(y)
            image_point[image_id] = torch.tensor([x, y])
    return image_point


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class WSOLImageLabelDataset(Dataset):
    def __init__(self, data_root, metadata_root, transform, proxy,
                 openimages_box, point_loc, box_loc, proxy_point, scales,
                 num_sample_per_class=0, multi_label=False, with_small=False,
                 small_count=100, **kwargs):
        self.point_loc = point_loc
        self.box_loc = box_loc
        self.multi_label = multi_label
        self.scales = scales

        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root, openimages_box,
                                           self.point_loc, proxy_point,
                                           box_loc=box_loc, with_small=with_small,
                                           small_count=small_count)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy,
                                       proxy_point=proxy_point)
        self.image_labels = get_class_labels(self.metadata, multi_label=self.multi_label)
        if point_loc:
            self.image_point = get_point_loc(self.metadata)
        elif box_loc:
            self.image_size = get_image_sizes(self.metadata)
            self.image_box = get_bounding_boxes(self.metadata)
        self.num_sample_per_class = num_sample_per_class
        self.img_normal = TorchvisionNormalize()

        # small aug
        self.smallaug = kwargs['smallaug']
        self.salmap_root = kwargs['salmap_root']
        self.target_transform = kwargs['target_transform'][0]
        self.split = kwargs['split']
        self._adjust_samples_per_class()

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        seed = np.random.randint(2147483647)
        if self.multi_label:
            # for PASCAL VOC
            if self.scales is not None:
                ms_img_list = []
                for s in self.scales:
                    s_image = image if s == 1 else pil_rescale(image, s, order=3)
                    s_image = self.img_normal(s_image)
                    s_image = np.transpose(s_image, (2, 0, 1))
                    ms_img_list.append(s_image)
                image = ms_img_list
            else:
                if self.smallaug:
                    set_random_seed(seed)
                image = self.transform(image)
            onehot_label = np.zeros(20)
            onehot_label[np.array(image_label, dtype=np.int)] = 1
            image_label = onehot_label
        else:
            if self.smallaug:
                set_random_seed(seed)
            image = self.transform(image)
        if self.point_loc:
            image_point = self.image_point[image_id]
            return image, image_label, image_id, image_point
        elif self.box_loc:
            x0, y0, x1, y1 = resize_bbox(self.image_box[image_id][0],
                                         self.image_size[image_id],
                                         (224, 224))
            w = x1 - x0
            h = y1 - y0
            x0, y0, w, h = square_crop_function((x0, y0, w, h), 224, 224)
            image_box = torch.tensor([x0, y0, w, h])
            return image, image_label, image_id, image_box
        elif self.smallaug and self.split == 'train':
            salmap_id = os.path.splitext(image_id)[0] + ".png"
            image_salmap = Image.open(os.path.join(self.salmap_root, salmap_id))
            set_random_seed(seed)
            image_salmap = self.target_transform(image_salmap)
            image_salmap = torch.where(image_salmap > 0.5, 1, 0)
            return image, image_label, image_id, image_salmap
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


def get_data_loader(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set, proxy_point, box_loc, dataset_name,
                    scales=None, openimages_box=False, point_loc=False, multi_label=False,
                    num_val_sample_per_class=0, with_small=False, small_count=100, **kwargs):
    kwargs['crop_size'] = crop_size
    if dataset_name != 'CUB':
        _SPLITS = ('train', 'val', 'test')
    else:
        _SPLITS = ('train', 'val', 'test', 'small')
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        small=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]))

    target_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),

    loaders = {
        split: DataLoader(
            WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0),
                openimages_box=openimages_box,
                point_loc=point_loc if split == 'train' else False,
                proxy_point=proxy_point if split == 'train' else False,
                box_loc=box_loc,
                multi_label=multi_label,
                scales=scales,
                with_small=with_small,
                small_count=small_count,
                split=split,
                target_transform=target_transform,
                **kwargs
            ),
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=workers)
        for split in _SPLITS
    }
    return loaders

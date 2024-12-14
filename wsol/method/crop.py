import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CropCAM']
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


class CropCAM(object):
    def __init__(self, large_feature_map, original_feature_map, **kwargs):
        self.input_size = (224, 224)
        self.output_size = (28, 28) if large_feature_map else (14, 14)
        if original_feature_map:
            self.output_size = (7, 7)
        architecture = kwargs['architecture']
        self.inception = True if architecture == "inception_v3" else False
            
        self.crop_threshold = kwargs['crop_threshold']
        self.crop_ratio = kwargs['crop_ratio']
        self.attention_cam = kwargs['attention_cam']

        self.loss_ratio = kwargs['loss_ratio']
        self.loss_pos = kwargs['loss_pos']
        self.loss_neg = kwargs['loss_neg']
        self.wsol_method = kwargs['wsol_method']
        self.other_method_loss_ratio = kwargs['other_method_loss_ratio']
        self.crop_method_loss_ratio = kwargs['crop_method_loss_ratio']

        # For several CAM normalization methods.
        self.norm_method = kwargs['norm_method']
        self.percentile = kwargs['percentile']
        self.crop_with_norm = kwargs['crop_with_norm']


        self.other_method_loss = torch.tensor(0)
        if self.wsol_method == "da":
            self.calculate_other_method_loss = True
        else:
            self.calculate_other_method_loss = False

        self.sigmoid_for_cam = torch.nn.Sigmoid()

    def normalize_tensor(self, x):
        channel_vector = x.view(x.size()[0], x.size()[1], -1)
        minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
        maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
        if not self.crop_with_norm or self.norm_method == 'minmax':
            minimum = torch.where(maximum - minimum > 0, minimum, 
                                  torch.tensor(0, dtype=torch.float, device='cuda'))
            normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
        elif self.norm_method == 'max':
            channel_vector = torch.where(channel_vector > 0, channel_vector, 
                                         torch.tensor(0, dtype=torch.float, device='cuda'))
            normalized_vector = torch.div(channel_vector, maximum)
        elif self.norm_method == 'pas':
            maximum = torch.quantile(channel_vector, torch.tensor(0.9, dtype=torch.float, device='cuda'),
                                     dim=-1, keepdim=True, interpolation='nearest')
            normalized_vector = torch.div(channel_vector - minimum, maximum)
            normalized_vector = torch.where(normalized_vector < 1, normalized_vector,
                                            torch.tensor(1, dtype=torch.float, device='cuda'))
        elif self.norm_method == 'ivr':
            minimum = torch.quantile(channel_vector, 
                                     torch.tensor(self.percentile, dtype=torch.float, device='cuda'),
                                     dim=-1, keepdim=True, interpolation='nearest')
            channel_vector -= minimum
            channel_vector = torch.where(channel_vector > 0, channel_vector, 
                                         torch.tensor(0, dtype=torch.float, device='cuda'))
            normalized_vector = torch.div(channel_vector, maximum)

        normalized_tensor = normalized_vector.view(x.size())
        return normalized_tensor

    def upsample_module(self, img, upsample_size, norm=False):
        img = F.interpolate(img, size=upsample_size,
                            mode='bicubic', align_corners=False)
        if norm:
            return self.normalize_tensor(img)
        return img

    def extract_cam(self, feature, cam_weights=None, original_cam=False,
                    inception=False, labels=None):
        if self.attention_cam:
            cams = feature.mean(1, keepdim=True)
        elif inception:
            batch_size = feature.shape[0]
            cams = feature[range(batch_size), labels]
            cams = cams.unsqueeze(dim=1)
        else:
            cams = (cam_weights.view(*feature.shape[:2], 1, 1) *
                    feature).mean(1, keepdim=True)
        if original_cam:
            cams = self.normalize_tensor(cams)
        return cams

    @staticmethod
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

    @staticmethod
    def random_select_range(crop_range):
        return crop_range[np.random.randint(0, crop_range.shape[0], size=1)[0]]

    def compute_crop_range(self, cam, train=True):
        cam_h, cam_w = cam.shape[2:]
        attentions = t2n(cam)
        crop_range = []
        for attention in attentions:
            _, gray_heatmap = cv2.threshold(
                src=attention,
                thresh=self.crop_threshold * np.max(attention),
                maxval=255,
                type=cv2.THRESH_BINARY)
            gray_heatmap = gray_heatmap.astype(np.uint8).reshape(cam_h, cam_w, 1)
            contours = cv2.findContours(
                image=gray_heatmap,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

            if len(contours) == 0:
                x, y, w, h = 0, 0, cam_w, cam_h
                crop_range.append([x, y, w, h])
                continue
            else:
                x, y, w, h = cam_w, cam_h, 0, 0

            for contour in contours:
                sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(contour)
                x = sub_x if sub_x < x else x
                y = sub_y if sub_y < y else y
                w = sub_x + sub_w - x if sub_x + sub_w > x + w else w
                h = sub_y + sub_h - y if sub_y + sub_h > y + h else h

            x, y, w, h = self.square_crop_function([x, y, w, h], cam_w, cam_h)

            crop_range.append([x, y, w, h])
        return np.array(crop_range)

    def crop_and_resize(self, input_img, crop_range):
        img_list = []
        # 224(input_img) / 28(cam) = 8
        resize_length = int(self.input_size[0] / self.output_size[0])
        crop_range_resize = crop_range * resize_length
        for img, (x, y, w, h) in zip(input_img, crop_range_resize):
            img = img[:, y:y+h, x:x+w].unsqueeze(dim=0)
            img = self.upsample_module(img, upsample_size=self.input_size)
            img_list.append(img.squeeze(dim=0))
        img_tensor = torch.stack(img_list, dim=0)
        return img_tensor

    def crop_to_original_and_bg(self, cam_org, cam_crop, crop_range):
        cam_org_map = cam_org.detach().clone()
        cam_crop_map = cam_crop.detach()
        img_list, bg_list = [], []
        crop_range = crop_range * int(self.input_size[0] / self.output_size[0])

        for img_org, img_crop, (x, y, w, h) in zip(cam_org_map, cam_crop_map, crop_range):
            # For negative value (Change sequence of normalization)
            img_crop = self.upsample_module(img_crop.unsqueeze(dim=0),
                                            upsample_size=(h, w), norm=True)

            img_org[:, y:y + h, x:x + w] = img_crop.squeeze()
            img_list.append(img_org)
            bg_list.append(1 - img_org)
        img_tensor = torch.stack(img_list, dim=0)
        bg_tensor = torch.stack(bg_list, dim=0)
        return img_tensor, bg_tensor

    def ratio_cam(self, cam_org, cam_crop, crop_range, crop_ratio):
        img_size = self.output_size[0] * self.output_size[1]
        img_list = []
        for img_org, img_crop, (x, y, w, h) in zip(cam_org, cam_crop, crop_range):
            if (w * h) / img_size < crop_ratio:
                img_list.append(img_crop)
            else:
                img_list.append(img_org)
        cams = torch.stack(img_list, dim=0)
        return cams

    def forward(self, model, images, target, return_cam=False):
        train = False if return_cam else True
        crop_images = images.detach().clone()
        output_dict = model(images, target, crop=True)
        crop_images, cam_org, crop_range, target = self.original_stage(crop_images, target,
                                                                       output_dict["feature_map"],
                                                                       output_dict["cam_weights"],
                                                                       train=train)
        output_dict_crop = model(crop_images, target, crop=True)
        if self.calculate_other_method_loss and not return_cam:
            self.other_method_loss = output_dict['loss'] + output_dict_crop['loss']

        cam_org, cam_crop, cam_bg = self.crop_stage(target,
                                                    output_dict_crop["feature_map"],
                                                    output_dict_crop["cam_weights"],
                                                    cam_org, crop_range, return_cam)

        output_dict_total = {
            'logits': output_dict["logits"], 'logits_crop': output_dict_crop["logits"],
            'attention': cam_org, 'attention_crop': cam_crop,
            'attention_bg': cam_bg,
            'crop_range': torch.from_numpy(crop_range).to('cuda'),
            'output_size': torch.Tensor(list(self.output_size)).to('cuda')
        }

        if return_cam:
            cam = self.ratio_cam(cam_org, cam_crop, crop_range, self.crop_ratio).squeeze(dim=1)
            return cam, output_dict["logits"]

        return output_dict_total

    def original_stage(self, crop_images, target, feature_map, cam_weights, train=True):
        cam_org = self.extract_cam(feature_map, cam_weights,
                                   original_cam=True, inception=self.inception, labels=target)
        crop_range = self.compute_crop_range(cam=cam_org, train=train)
        crop_images = self.crop_and_resize(crop_images, crop_range)
        crop_images = crop_images.requires_grad_(True)
        return crop_images, cam_org, crop_range, target

    def crop_stage(self, target, feature_map, cam_weights, cam_org, crop_range, return_cam=False):
        cam_crop = self.extract_cam(feature_map, cam_weights,
                                    original_cam=False, inception=self.inception, labels=target)
        if return_cam:
            cam_org = self.upsample_module(cam_org, upsample_size=self.input_size, norm=True)
        cam_crop, cam_bg = self.crop_to_original_and_bg(cam_org, cam_crop, crop_range)
        return cam_org, cam_crop, cam_bg

    # Get Contrastive Loss
    def get_loss(self, output_dict, target=0):
        loss_exp = LossExperiment(output_dict, target)
        att_loss = loss_exp.att_loss(self.loss_pos, self.loss_neg)

        cls_loss = loss_exp.cls_loss()
        total_loss = (2 * self.loss_ratio) * cls_loss + (2 * (1 - self.loss_ratio)) * att_loss
        if self.calculate_other_method_loss:
            return self.other_method_loss_ratio * self.other_method_loss + \
                   self.crop_method_loss_ratio * total_loss, att_loss, cls_loss
        return total_loss, att_loss, cls_loss


class LossExperiment(object):
    def __init__(self, output_dict, target):
        cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.loss_cls_org = cross_entropy_loss(output_dict['logits'], target)
        self.loss_cls_crop = cross_entropy_loss(output_dict['logits_crop'], target)
        self.target = target

        self.cam_org = output_dict['attention']
        self.cam_crop = output_dict['attention_crop']
        self.cam_bg = output_dict['attention_bg']

    def cls_loss(self):
        return 0.5 * self.loss_cls_org + 0.5 * self.loss_cls_crop

    def att_loss(self, loss_pos, loss_neg):
        pos = 1 - torch.mean(self.cam_org * self.cam_crop) / \
              ((torch.mean(self.cam_org ** 2)) ** (1 / 2) * (torch.mean(self.cam_crop ** 2)) ** (1 / 2))
        neg = 1 - torch.mean(self.cam_org * self.cam_bg) / \
              ((torch.mean(self.cam_org ** 2)) ** (1 / 2) * (torch.mean(self.cam_bg ** 2)) ** (1 / 2))
        return loss_pos * pos - loss_neg * neg + 1




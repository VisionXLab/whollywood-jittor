import jittor as jt
from jittor import nn
from jdet.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS, NECKS
import math
from jittor.nn import grid_sample
import cv2
import copy
import numpy as np


def plot_one_rotated_box(img,
                         obb,
                         color=[0.0, 0.0, 128],
                         label=None,
                         line_thickness=None):
    width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
    if theta < 0:
        width, height, theta = height, width, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(
        cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.drawContours(
        image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)
    c1 = (int(obb[0]), int(obb[1]))
    if label:
        tl = 2
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        textcolor = [0, 0, 0] if max(color) > 192 else [255, 255, 255]
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, textcolor, thickness=tf, lineType=cv2.LINE_AA)


@MODELS.register_module()
class H2RBoxV2P(nn.Module):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 prob_rot=0.95,
                 view_range=(0.25, 0.75)):
        super(H2RBoxV2P, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS)
        else:
            self.neck = None
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.prob_rot = prob_rot
        self.view_range = view_range

    def rotate(self, img, theta):
        n, c, h, w = img.shape
        cosa, sina = math.cos(theta), math.sin(theta)
        tf = jt.array([[cosa, -sina], [sina, cosa]], dtype=jt.float)
        x_range = jt.linspace(-1, 1, w)
        y_range = jt.linspace(-1, 1, h)
        y, x = jt.meshgrid(y_range, x_range)
        grid = jt.stack([x, y], -1).unsqueeze(0).expand([n, -1, -1, -1])
        grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
        img = grid_sample(img, grid, 'bilinear', 'reflection',
                            align_corners=True)
        return img

    def vflip(self, img):
        n, c, h, w = img.shape
        tf = jt.array([[1, 0], [0, -1]], dtype=jt.float)
        x_range = jt.linspace(-1, 1, w)
        y_range = jt.linspace(-1, 1, h)
        y, x = jt.meshgrid(y_range, x_range)
        grid = jt.stack([x, y], -1).unsqueeze(0).expand([n, -1, -1, -1])
        grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
        img = grid_sample(img, grid, 'bilinear', 'reflection',
                            align_corners=True)
        return img

    def forward_train(self, images, targets):
        gt_bboxes = [target['rboxes'] for target in targets]

        # Add an id to each annotation to match objects in different views
        offset = 1
        for i, bboxes in enumerate(gt_bboxes):
            bids = jt.arange(
                0, len(bboxes), 1) + offset
            gt_bboxes[i] = jt.cat((bboxes, bids[:, None]), dim=-1)
            offset += len(bboxes)

        # Concat original/rotated/flipped images and gts in the batch dim
        if jt.rand(1) < self.prob_rot:
            rot = math.pi * (
                jt.rand(1).item() *
                (self.view_range[1] - self.view_range[0]) + self.view_range[0])
            img_ss = self.rotate(images, rot)
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = jt.array([[cosa, -sina], [sina, cosa]], dtype=jt.float)
            ctr = jt.array([[images.shape[-1] / 2, images.shape[-2] / 2]], dtype=jt.float)
            gt_bboxes_ss = copy.deepcopy(gt_bboxes)
            for bboxes in gt_bboxes_ss:
                bboxes[:, :2] = (bboxes[..., :2] - ctr).matmul(tf.transpose()) + ctr
                bboxes[:, 4] = bboxes[:, 4] + rot
                bboxes[:, 5] = bboxes[:, 5] + 0.5

            images = jt.cat((images, img_ss), 0)
            gt_bboxes = gt_bboxes + gt_bboxes_ss
            gt_sstype = ('rot', rot)
        else:
            img_ss = self.vflip(images)
            gt_bboxes_ss = copy.deepcopy(gt_bboxes)
            for bboxes in gt_bboxes_ss:
                bboxes[:, 1] = images.shape[-2] - bboxes[:, 1]
                bboxes[:, 4] = -bboxes[:, 4]
                bboxes[:, 5] = bboxes[:, 5] + 0.5

            images = jt.cat((images, img_ss), 0)
            gt_bboxes = gt_bboxes + gt_bboxes_ss
            gt_sstype = ('flp', 0)
        
        targets = targets + copy.deepcopy(targets)
        for i, target in enumerate(targets):
            target['rboxes'] = gt_bboxes[i]
            target['ss'] = gt_sstype

        if False:
            idx = np.random.randint(100)
            B = len(images)
            for i in range(B):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img[..., (2, 1, 0)] * 58 + 127)
                bb = targets[i]['rboxes']
                ll = targets[i]['labels']
                bb[bb[..., 2] < 3, 2] = 3
                bb[bb[..., 3] < 3, 3] = 3
                for b, l in zip(bb.cpu().numpy(), ll.cpu().numpy()):
                    plot_one_rotated_box(img, b, label=f'{l}')
                cv2.imwrite(f'{idx}-{i}.png', img)

        x = self.backbone(images)
        if self.neck:
            x = self.neck(x)
        losses = self.bbox_head.execute_train(x, targets)  
        return losses

    def forward_test(self, images, targets):
        feat = self.backbone(images)
        if self.neck:
            feat = self.neck(feat)
        outs = self.bbox_head.forward(feat)
        return self.bbox_head.get_bboxes(*outs, targets)

    def execute(self, img, targets):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if 'rboxes' in targets[0]:
            return self.forward_train(img, targets)
        else:
            return self.forward_test(img, targets)

import jittor as jt
from jittor import nn
from jdet.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS, NECKS
from jdet.models.boxes.box_ops import rotated_box_to_bbox, boxes_x0y0x1y1_to_xywh
from jdet.models.networks.h2rbox_v2p import plot_one_rotated_box
import math
from jittor.nn import grid_sample
import cv2
import copy
import numpy as np

from jdet.models.synthesis_generators import point2rbox_generator


@MODELS.register_module()
class P2RSubnet(nn.Module):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 basic_pattern: str = 'data/dota',
                 sca_fact: float = 1.0,
                 warmup: int = 1000,
                 dense_cls: list = [],
                 square_cls: list = [],
                 use_synthesis: bool = True,
                 use_setrc: bool = True,
                 use_setsk: bool = True,
                 debug: bool = False):
        super(P2RSubnet, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS)
        else:
            self.neck = None
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.basic_pattern = basic_pattern
        self.sca_fact = sca_fact
        self.warmup = warmup
        self.dense_cls = dense_cls
        self.square_cls = square_cls
        self.use_synthesis = use_synthesis
        self.debug = debug
        self.basic_pattern = point2rbox_generator.load_basic_pattern(
            self.basic_pattern, use_setrc, use_setsk)

    def add_synthesis(self, images, targets):

        def synthesis_single(img, bboxes, labels):
            labels = labels[:, None] - 1
            bb = jt.cat((bboxes, jt.ones_like(labels), labels), -1)
            img, bb = point2rbox_generator.generate_sythesis(
                img, bb, self.sca_fact, *self.basic_pattern, self.dense_cls,
                img.shape[-1])
            labels =  bb[:, 6].long()
            square_mask = jt.zeros_like(labels, dtype=jt.bool)
            for c in self.square_cls:
                square_mask = jt.logical_or(square_mask, labels == c)
            bb[square_mask, 4] = 0
            bboxes = jt.cat((boxes_x0y0x1y1_to_xywh(rotated_box_to_bbox(bb[:, :5])),
                             jt.zeros_like(bb[:, 5:6])), -1)
            return img, bboxes, labels + 1

        p = ((synthesis_single)(img, tar['rboxes'], tar['labels'])
             for (img, tar) in zip(images, targets))

        img, bboxes, labels = zip(*p)
        images = jt.stack(img, 0).to(images)
        for i, target in enumerate(targets):
            target['rboxes'] = jt.cat((target['rboxes'], bboxes[i]), 0)
            target['labels'] = jt.cat((target['labels'], labels[i]), 0)
        
        return images, targets

    def forward_train(self, images, targets):
        # Generate synthetic objects
        if self.use_synthesis:
            images, targets_syn = self.add_synthesis(images, copy.deepcopy(targets))
            
        x = self.backbone(images)
        if self.neck:
            x = self.neck(x)
        losses = self.bbox_head.execute_train(x, targets_syn)  
        targets = self.bbox_head.predict(x, targets)
        
        if self.debug:
            idx = np.random.randint(100)
            B = len(images)
            for i in range(B):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img[..., (2, 1, 0)] * 58 + 127)
                bb = targets_syn[i]['rboxes']
                ll = targets_syn[i]['labels']
                bb[bb[..., 2] < 3, 2] = 3
                bb[bb[..., 3] < 3, 3] = 3
                for b, l in zip(bb.cpu().numpy(), ll.cpu().numpy()):
                    plot_one_rotated_box(img, b, label=f'{l}')
                bb = targets[i]['rboxes']
                ll = targets[i]['labels']
                bb[bb[..., 2] < 3, 2] = 3
                bb[bb[..., 3] < 3, 3] = 3
                for b, l in zip(bb.cpu().numpy(), ll.cpu().numpy()):
                    plot_one_rotated_box(img, b, label=f'{l}', color=[0, 255, 0])
                cv2.imwrite(f'{idx}-{i}.png', img)

        return losses, targets

@MODELS.register_module()
class WhollyWood(nn.Module):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 p2rsubnet,
                 prob_rot=0.95,
                 view_range=(0.25, 0.75)):
        super(WhollyWood, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS)
        else:
            self.neck = None
        self.bbox_head = build_from_cfg(bbox_head, HEADS)
        self.p2rsubnet = build_from_cfg(p2rsubnet, MODELS) if p2rsubnet else None

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
        # P2RSubnet training
        if self.p2rsubnet:
            losses_p2r, targets = self.p2rsubnet.forward_train(
                images, targets)

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
        if self.p2rsubnet:
            if self.p2rsubnet.warmup > 0:
                self.p2rsubnet.warmup -= 1
                for k in list(losses):
                    losses[k] *= 0
            for k, v in losses_p2r.items():
                losses[k + '_p2r'] = v
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

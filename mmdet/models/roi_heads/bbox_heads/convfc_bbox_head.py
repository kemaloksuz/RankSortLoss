import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.core import vectorize_labels, bbox_overlaps, force_fp32, multiclass_nms
from mmcv.ops.nms import batched_nms

from mmdet.models.losses import ranking_losses
import numpy as np
import collections
from scipy import stats
from mmdet.models.losses import accuracy

import pdb

@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule 
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@HEADS.register_module()
class RankBasedShared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, rank_loss_type = 'RankSort', *args, **kwargs):
        super(RankBasedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        self.rank_loss_type = rank_loss_type
        if self.rank_loss_type == 'RankSort':
            self.loss_rank = ranking_losses.RankSort()
        elif self.rank_loss_type == 'aLRP':
            self.loss_rank = ranking_losses.aLRPLoss()
            self.SB_weight = 50
            self.period = 7330
            self.cls_LRP_hist = collections.deque(maxlen=self.period)
            self.reg_LRP_hist = collections.deque(maxlen=self.period)
            self.counter = 0

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        flat_labels = vectorize_labels(labels, self.num_classes, label_weights)
        flat_preds = cls_score.reshape(-1)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                pos_target = bbox_targets[pos_inds]
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds,labels[pos_inds]]

                loss_bbox = self.loss_bbox(pos_bbox_pred, pos_target)

                if self.rank_loss_type == 'RankSort':

                    bbox_weights = cls_score.detach().sigmoid().max(dim=1)[0][pos_inds]

                    IoU_targets = bbox_overlaps(pos_bbox_pred.detach(), pos_target, is_aligned=True)
                    flat_labels[flat_labels==1] = IoU_targets

                    ranking_loss, sorting_loss = self.loss_rank.apply(flat_preds, flat_labels)

                    bbox_avg_factor = torch.sum(bbox_weights)
                    if bbox_avg_factor < 1e-10:
                        bbox_avg_factor = 1
                
                    losses_bbox = torch.sum(bbox_weights*loss_bbox)/bbox_avg_factor
                    self.SB_weight = (ranking_loss+sorting_loss).detach()/float(losses_bbox.item())
                    losses_bbox *= self.SB_weight
                    return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox), bbox_weights

                elif self.rank_loss_type == 'aLRP':
                    losses_cls, rank, order = self.loss_rank.apply(flat_preds, flat_labels, loss_bbox.detach())
                    
                    # Order the regression losses considering the scores. 
                    ordered_losses_bbox = loss_bbox[order.detach()].flip(dims=[0])
            
                    # aLRP Regression Component
                    losses_bbox = ((torch.cumsum(ordered_losses_bbox,dim=0)/rank[order.detach()].detach().flip(dims=[0])).mean())

                    # Self-balancing
                    self.cls_LRP_hist.append(float(losses_cls.item()))
                    self.reg_LRP_hist.append(float(losses_bbox.item()))
                    self.counter+=1
                
                    if self.counter == self.period:
                        self.SB_weight = (np.mean(self.reg_LRP_hist)+np.mean(self.cls_LRP_hist))/np.mean(self.reg_LRP_hist)
                        self.cls_LRP_hist.clear()
                        self.reg_LRP_hist.clear()
                        self.counter=0
                    losses_bbox *= self.SB_weight

                    return dict(loss_cls=losses_cls, loss_bbox=losses_bbox), None
            else:
                losses_bbox=bbox_pred.sum()*0+1
                if self.rank_loss_type == 'RankSort':
                    ranking_loss = cls_score.sum()*0+1
                    sorting_loss = cls_score.sum()*0+1
                    return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox), bbox_weights
                else:
                    losses_cls = cls_score.sum()*0+1
                    return dict(loss_cls=losses_cls, loss_bbox=losses_bbox), None

        else:
            losses_bbox=bbox_pred.sum()*0+1
            if self.rank_loss_type == 'RankSort':
                ranking_loss = cls_score.sum()*0+1
                sorting_loss = cls_score.sum()*0+1
                return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox), bbox_weights
            else:
                losses_cls = cls_score.sum()*0+1
                return dict(loss_cls=losses_cls, loss_bbox=losses_bbox), None

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        scores = cls_score.sigmoid() if cls_score is not None else None

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
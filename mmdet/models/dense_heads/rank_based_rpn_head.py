import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmdet.core import vectorize_labels, bbox_overlaps
from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin
import numpy as np
import collections

from mmdet.models.losses import ranking_losses
import pdb

@HEADS.register_module()
class RankBasedRPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, head_weight=0.20, rank_loss_type = 'RankSort', **kwargs):
        super(RankBasedRPNHead, self).__init__(1, in_channels, **kwargs)
        self.head_weight = head_weight
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

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    '''

    def flatten_labels(self, flat_labels, label_weights):
        prediction_number = flat_labels.shape[0]
        labels = torch.zeros( [prediction_number], device=flat_labels.device)
        labels[flat_labels == 0] = 1.
        labels[label_weights == 0] = -1.
        return labels.reshape(-1)
    '''
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        all_labels=[]
        all_label_weights=[]
        all_cls_scores=[]
        all_bbox_targets=[]
        all_bbox_weights=[]
        all_bbox_preds=[]
        for labels, label_weights, cls_score, bbox_targets, bbox_weights, bbox_pred in zip(labels_list, label_weights_list,cls_scores, bbox_targets_list, bbox_weights_list, bbox_preds):
            all_labels.append(labels.reshape(-1))
            all_label_weights.append(label_weights.reshape(-1))
            all_cls_scores.append(cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))
            
            all_bbox_targets.append(bbox_targets.reshape(-1, 4))
            all_bbox_weights.append(bbox_weights.reshape(-1, 4))
            all_bbox_preds.append(bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4))

        cls_labels = torch.cat(all_labels)
        all_scores=torch.cat(all_cls_scores)
        pos_idx = (cls_labels < self.num_classes)
        #flatten_anchors = torch.cat([torch.cat(item, 0) for item in anchor_list])
        if pos_idx.sum() > 0:
            # regression loss
            pos_pred = self.delta2bbox(torch.cat(all_bbox_preds)[pos_idx])
            pos_target = self.delta2bbox(torch.cat(all_bbox_targets)[pos_idx])
            loss_bbox = self.loss_bbox(pos_pred, pos_target)

            # flat_labels = self.flatten_labels(cls_labels, torch.cat(all_label_weights))
            flat_labels = vectorize_labels(cls_labels, self.num_classes, torch.cat(all_label_weights))
            flat_preds = all_scores.reshape(-1)
            if self.rank_loss_type == 'RankSort':
                pos_weights = all_scores.detach().sigmoid().max(dim=1)[0][pos_idx]

                bbox_avg_factor = torch.sum(pos_weights)
                if bbox_avg_factor < 1e-10:
                    bbox_avg_factor = 1

                loss_bbox = torch.sum(pos_weights*loss_bbox)/bbox_avg_factor

                IoU_targets = bbox_overlaps(pos_pred.detach(), pos_target, is_aligned=True)
                flat_labels[flat_labels==1]=IoU_targets
                ranking_loss, sorting_loss = self.loss_rank.apply(flat_preds, flat_labels)

                self.SB_weight = (ranking_loss+sorting_loss).detach()/float(loss_bbox.item())
                loss_bbox *= self.SB_weight

                return dict(loss_rpn_rank=self.head_weight*ranking_loss, loss_rpn_sort=self.head_weight*sorting_loss, loss_rpn_bbox=self.head_weight*loss_bbox)

            elif self.rank_loss_type == 'aLRP':
                e_loc = loss_bbox.detach()/(2*(1-0.7))
                losses_cls, rank, order = self.loss_rank.apply(flat_preds, flat_labels, e_loc)
                
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
                return dict(loss_rpn_cls=self.head_weight*losses_cls, loss_rpn_bbox=self.head_weight*losses_bbox)

        else:
            losses_bbox=torch.cat(all_bbox_preds).sum()*0+1
            if self.rank_loss_type == 'RankSort':
                ranking_loss = all_scores.sum()*0+1
                sorting_loss = all_scores.sum()*0+1
                return dict(loss_rpn_rank=self.head_weight*ranking_loss, loss_rpn_sort=self.head_weight*sorting_loss, loss_rpn_bbox=self.head_weight*losses_bbox)
            else:
                losses_cls = all_scores.sum()*0+1
                return dict(loss_rpn_cls=self.head_weight*losses_cls, loss_rpn_bbox=self.head_weight*losses_bbox)

    def delta2bbox(self, deltas, means=[0., 0., 0., 0.], stds=[0.1, 0.1, 0.2, 0.2], max_shape=None, wh_ratio_clip=16/1000):

        wx, wy, ww, wh = stds
        dx = deltas[:, 0] * wx
        dy = deltas[:, 1] * wy
        dw = deltas[:, 2] * ww
        dh = deltas[:, 3] * wh
        
        max_ratio = np.abs(np.log(wh_ratio_clip))

        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = torch.exp(dw)
        pred_h = torch.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]

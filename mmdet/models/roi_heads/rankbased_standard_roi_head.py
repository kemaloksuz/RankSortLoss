import torch
import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
import pdb

@HEADS.register_module()
class RankBasedStandardRoIHead(StandardRoIHead):

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            cls_loss_val = bbox_results['loss_bbox']['loss_roi_rank'].detach()+bbox_results['loss_bbox']['loss_roi_sort'].detach()
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    bbox_results['bbox_weights'],
                                                    cls_loss_val,
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses


    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox, bbox_weights = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_weights=bbox_weights)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, bbox_weights, cls_loss_val,
                            gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            sampling_results, bbox_weights =self.resample_for_mask_head(sampling_results, bbox_weights)
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels, bbox_weights, cls_loss_val)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def resample_for_mask_head(self, sampling_results, bbox_weights, max_mask=200):
        num_img=len([res.pos_bboxes for res in sampling_results])
        pos_nums=np.zeros([num_img])
        for i,res in enumerate(sampling_results):
            pos_nums[i] = res.pos_bboxes.shape[0]

        all_pos = pos_nums.sum().astype(int)

        if all_pos <= max_mask:
            return sampling_results, bbox_weights
        else:
            # Select indices of positive masks
            idx = torch.randperm(all_pos.item())[:max_mask]

            bbox_weights_=bbox_weights[idx]

            # Distribute indices to sampling result
            min_idx = 0
            max_idx = 0
            for res in sampling_results:
                pos_num_i = res.pos_bboxes.shape[0]
                max_idx += pos_num_i
                idx_i = (idx >= min_idx) & (idx < max_idx) 
                idx_valid = idx[idx_i]-min_idx
                res.pos_assigned_gt_inds=res.pos_assigned_gt_inds[idx_valid]
                res.pos_bboxes=res.pos_bboxes[idx_valid]
                res.pos_inds=res.pos_inds[idx_valid]
                res.pos_is_gt=res.pos_is_gt[idx_valid]
                res.pos_gt_labels=res.pos_gt_labels[idx_valid]
                min_idx += pos_num_i

            return sampling_results, bbox_weights_




import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector
from mmdet.models.losses import ranking_losses

import pdb

@DETECTORS.register_module()
class RankSortYOLACT(SingleStageDetector):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RankSortYOLACT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)

        self.mask_head = build_head(mask_head)
        self.init_segm_mask_weights()
        self.loss_rank = ranking_losses.RankSort()

    def init_segm_mask_weights(self):
        """Initialize weights of the YOLACT semg head and YOLACT mask head."""
        self.mask_head.init_weights()

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
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
        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        x = self.extract_feat(img)

        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        loss_bbox, IoU_targets, pos_weights, flat_labels, flat_preds, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        mask_pred = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas,
                                   sampling_results)

        dice = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                        img_metas, sampling_results)

        flat_labels[flat_labels==1]=IoU_targets

        ranking_loss, sorting_loss = self.loss_rank.apply(flat_preds, flat_labels)

        loss_mask = 1-(torch.sum(pos_weights*dice)/(pos_weights.sum()))

        SB_box = (ranking_loss+sorting_loss).detach()/float(loss_bbox.item())
        loss_bbox *= SB_box

        SB_mask = (ranking_loss+sorting_loss).detach()/float(loss_mask.item())
        loss_mask *= SB_mask

        losses = dict(ranking_loss=ranking_loss,sorting_loss=sorting_loss,
                      loss_bbox=loss_bbox, loss_mask=loss_mask)

        # check NaN and Inf
        for loss_name in losses.keys():
            assert torch.isfinite(losses[loss_name])\
                .all().item(), '{} becomes infinite or NaN!'\
                .format(loss_name)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation."""
        x = self.extract_feat(img)

        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)

        bbox_inputs = (cls_score, bbox_pred,
                       coeff_pred) + (img_metas, self.test_cfg, rescale)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.get_bboxes(
            *bbox_inputs)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        num_imgs = len(img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_preds = self.mask_head(x[0], det_coeffs, _bboxes, img_metas)
            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], det_labels[i], img_metas[i], rescale)
                    segm_results.append(segm_result)
        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError

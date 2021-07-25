from mmcv.runner import auto_fp16, force_fp32
from .fcn_mask_head import FCNMaskHead
from mmdet.models.builder import HEADS
import torch
import pdb

@HEADS.register_module()
class RankBasedFCNMaskHead(FCNMaskHead):

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        dice = 2 * intersection / union
        return dice

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels, bbox_weights, cls_loss_val):
        loss = dict()
        num_rois = mask_pred.size()[0]
        if num_rois == 0:
            losses_mask = mask_pred.sum()
        else:
            inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
            pred_slice = mask_pred[inds, labels].squeeze(1)
            pred_slice = pred_slice.sigmoid()
            loss_mask = 1-self.dice_coefficient(pred_slice, mask_targets)
            
            bbox_avg_factor = torch.sum(bbox_weights)
            if bbox_avg_factor < 1e-10:
                bbox_avg_factor = 1
                
            losses_mask = torch.sum(bbox_weights*loss_mask)/bbox_avg_factor
            self.SB_weight = cls_loss_val.detach()/float(losses_mask.item())
            losses_mask *= self.SB_weight

        loss['loss_mask'] = losses_mask
        return loss

import torch
from torchmetrics import Metric
import torch.nn.functional as F


class SSCMetrics(Metric):

    def __init__(self, num_classes, ignore_index=255, gaze=False):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gaze = gaze

        for metric in ('tp_sc', 'fp_sc', 'fn_sc'):
            self.add_state(metric, torch.tensor(0), dist_reduce_fx='sum')

        for metric in ('tps_ssc', 'fps_ssc', 'fns_ssc'):
            self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

        
        if self.gaze:
            #inside gaze area
            for metric in ('tps_ssc_in', 'fps_ssc_in', 'fns_ssc_in'):
                self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

            #outside gaze area
            for metric in ('tps_ssc_out', 'fps_ssc_out', 'fns_ssc_out'): 
                self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds['ssc_logits'], dim=1)
        self.ins_indices = target['ins_cls_info']['indices'].squeeze(0).cpu().numpy().tolist()[1:]
        self.bk_indices = target['bk_cls_info']['indices'].squeeze(0).cpu().numpy().tolist()[1:]
        target_tensor = target['target']
        mask = target_tensor != self.ignore_index

        tp, fp, fn = self._calculate_sc_scores(preds, target_tensor, mask)
        self.tp_sc += tp
        self.fp_sc += fp
        self.fn_sc += fn

        tp, fp, fn = self._calculate_ssc_scores(preds, target_tensor, mask)
        self.tps_ssc += tp
        self.fps_ssc += fp
        self.fns_ssc += fn

        #gaze mask
        if self.gaze:
            gaze_3d= target['gaze_3d']
            gaze_3d = F.interpolate(
                    gaze_3d.unsqueeze(1).float(), 
                    size=target_tensor.shape[1:],  
                    mode='nearest'                 
                ).squeeze(1)
            gaze_mask = gaze_3d > 0.5

            #areas inside gaze mask
            mask_in = mask & gaze_mask
            tp_in, fp_in, fn_in = self._calculate_ssc_scores(preds, target_tensor, mask_in)
            self.tps_ssc_in += tp_in
            self.fps_ssc_in += fp_in
            self.fns_ssc_in += fn_in

            #areas outside gaze mask
            mask_out = mask & ~gaze_mask
            tp_out, fp_out, fn_out = self._calculate_ssc_scores(preds, target_tensor, mask_out)
            self.tps_ssc_out += tp_out
            self.fps_ssc_out += fp_out
            self.fns_ssc_out += fn_out
        


    def compute(self):
        if self.tp_sc != 0:
            precision = self.tp_sc / (self.tp_sc + self.fp_sc)
            recall = self.tp_sc / (self.tp_sc + self.fn_sc)
            iou = self.tp_sc / (self.tp_sc + self.fp_sc + self.fn_sc)
        else:
            precision, recall, iou = 0, 0, 0
        ious = self.tps_ssc / (self.tps_ssc + self.fps_ssc + self.fns_ssc + 1e-6)
        if self.gaze:
            ious_in = self.tps_ssc_in / (self.tps_ssc_in + self.fps_ssc_in + self.fns_ssc_in + 1e-6)
            ious_out = self.tps_ssc_out / (self.tps_ssc_out + self.fps_ssc_out + self.fns_ssc_out + 1e-6)

            return {
                'Precision': precision,
                'Recall': recall,
                'IoU': iou,
                'iou_per_class': ious,
                'iou_per_class_in': ious_in,
                'iou_per_class_out': ious_out,
                'mIoU': ious[1:].mean(),
                'InsmIoU': ious[self.ins_indices].mean(),
                'BkmIoU': ious[self.bk_indices].mean(),

                #gaze area
                'Gaze_mIoU': ious_in[1:].mean(),
                'Gaze_InsmIoU': ious_in[self.ins_indices].mean(),
                'Gaze_BkmIoU': ious_in[self.bk_indices].mean(),

                #outside gaze area
                'Gaze_mIoU_out': ious_out[1:].mean(),
                'Gaze_InsmIoU_out': ious_out[self.ins_indices].mean(),
                'Gaze_BkmIoU_out': ious_out[self.bk_indices].mean(),
            }
        else:
            return {
                'Precision': precision,
                'Recall': recall,
                'IoU': iou,
                'iou_per_class': ious,
                'mIoU': ious[1:].mean(),
                'InsmIoU': ious[self.ins_indices].mean(),
                'BkmIoU': ious[self.bk_indices].mean(),
            }

    def _calculate_sc_scores(self, preds, targets, nonempty=None):
        preds = preds.clone()
        targets = targets.clone()
        bs = preds.shape[0]

        mask = targets == self.ignore_index
        preds[mask] = 0
        targets[mask] = 0

        preds = preds.flatten(1)
        targets = targets.flatten(1)
        preds = torch.where(preds > 0, 1, 0)
        targets = torch.where(targets > 0, 1, 0)

        tp, fp, fn = 0, 0, 0
        for i in range(bs):
            pred = preds[i]
            target = targets[i]
            if nonempty is not None:
                nonempty_ = nonempty[i].flatten()
                pred = pred[nonempty_]
                target = target[nonempty_]
            pred = pred.bool()
            target = target.bool()

            tp += torch.logical_and(pred, target).sum()
            fp += torch.logical_and(pred, ~target).sum()
            fn += torch.logical_and(~pred, target).sum()
        return tp, fp, fn

    def _calculate_ssc_scores(self, preds, targets, nonempty=None):
        preds = preds.clone()
        targets = targets.clone()
        bs = preds.shape[0]
        C = self.num_classes

        mask = targets == self.ignore_index
        preds[mask] = 0
        targets[mask] = 0

        preds = preds.flatten(1)
        targets = targets.flatten(1)

        tp = torch.zeros(C, dtype=torch.int).to(preds.device)
        fp = torch.zeros(C, dtype=torch.int).to(preds.device)
        fn = torch.zeros(C, dtype=torch.int).to(preds.device)
        for i in range(bs):
            pred = preds[i]
            target = targets[i]
            if nonempty is not None:
                mask = nonempty[i].flatten() & (target != self.ignore_index)
                pred = pred[mask]
                target = target[mask]
            for c in range(C):
                tp[c] += torch.logical_and(pred == c, target == c).sum()
                fp[c] += torch.logical_and(pred == c, ~(target == c)).sum()
                fn[c] += torch.logical_and(~(pred == c), target == c).sum()
        return tp, fp, fn

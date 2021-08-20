import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def _expand_onehot_labels(labels, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = paddle.full((labels.shape[0], label_channels), 0)
    inds = paddle.nonzero(labels >= 1, as_tuple=False).squeeze()
    if paddle.numel(inds) > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    
    return bin_labels


def binary_cross_entropy(pred, label, use_sigmoid=False, weight=None, reduction='mean', class_weight=None):
    if pred.rank() != label.rank():
        label, weight = _expand_onehot_labels(label, weight, pred.shape[-1])
        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()
        if use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(
                pred, label.float(), weight=class_weight, reduction='none')
        else:
            loss = F.binary_cross_entropy(
                pred, label.float(), weight=class_weight, reduction='none')
        # do the reduction for the weighted loss
    
    return loss


class AffinityLoss(nn.Layer):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = binary_cross_entropy
    
    def forward(self, cls_score, label):
        unary_term = self.cls_criterion(cls_score, label)
        
        diagonal_matrix = (1 - paddle.eye(label.shape[1]))
        vtarget = diagonal_matrix * label
        
        recall_part = paddle.sum(cls_score * vtarget, axis=2)
        denominator = paddle.sum(vtarget, axis=2)
        denominator = denominator <= 0
        recall_part = recall_part * denominator
        recall_label = paddle.ones_like(recall_part)
        recall_loss = self.cls_criterion(recall_part, recall_label)
        
        spec_part = paddle.sum((1 - cls_score) * (1 - label), axis=2)
        denominator = paddle.sum(1 - label, axis=2)
        denominator = denominator <= 0
        spec_part = spec_part * denominator
        spec_label = paddle.ones_like(spec_part)
        spec_loss = self.cls_criterion(
            spec_part,
            spec_label,
        )
        
        precision_part = paddle.sum(cls_score * vtarget, axis=2)
        denominator = paddle.sum(cls_score, axis=2)
        denominator = denominator <= 0
        precision_part = precision_part * denominator
        precision_label = paddle.ones_like(precision_part)
        precision_loss = F.binary_cross_entropy(precision_part, precision_label)
        
        global_term = recall_loss + spec_loss + precision_loss
        loss_cls = self.loss_weight * (unary_term + global_term)
        return loss_cls

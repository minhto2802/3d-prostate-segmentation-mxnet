# coding=utf-8
import numpy as np
from mxnet import gluon, nd


class DiceLoss(gluon.loss.Loss):
    """correlation loss"""

    def __init__(self, axis=[0, 1], weight=1., batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._batch_axis = batch_axis
        self.smooth = 1.

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Forward"""
        label = nd.one_hot(label, depth=2).transpose((0, 4, 1, 2, 3))
        intersection = F.sum(label * pred, axis=self._axis, exclude=True)
        union = F.sum(label + pred, axis=self._axis, exclude=True)
        dice = (2.0 * F.sum(intersection, axis=1) + self.smooth) / (F.sum(union, axis=1) + self.smooth)
        # return F.log(1 - dice)
        # return 1 - dice
        # return F.exp(-dice)
        # return 1 - dice.sum() / np.prod(dice.shape)
        return -dice

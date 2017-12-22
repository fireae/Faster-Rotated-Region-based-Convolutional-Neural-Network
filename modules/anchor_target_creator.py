import numpy as np

import chainer
from chainer import cuda

from modules.bbox2loc import bbox2loc
from modules.bbox_iou import bbox_iou
from modules.angle_grid import angle_grid
import math

class AnchorTargetCreator(object):

    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of obj

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`W, H`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """
        xp = cuda.get_array_module(bbox)
        bbox = cuda.to_cpu(bbox)
        anchor = cuda.to_cpu(anchor)

        img_W, img_H = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_W, img_H)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        if xp != np:
            loc = chainer.cuda.to_gpu(loc)
            label = chainer.cuda.to_gpu(label)
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index), ), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index, angle_thres = math.pi/12)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        # print len(pos_index)
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        # print len(neg_index)
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index, angle_thres):
        xp = cuda.get_array_module(anchor)
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        angle_gr = angle_grid(anchor,bbox)
        secondary_ious = xp.copy(ious)
        secondary_ious[angle_gr>angle_thres] = 0
        # ious[(ious>0.7)&(angle_gr>math.pi/16)] = 0
        # index of maximum iou for each anchor in each ROW
        argmax_ious = secondary_ious.argmax(axis=1)
        ious[(ious > 0.7) & (angle_gr > angle_thres)] = 0
        # the corresponding iou of that maximum index of anchor
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        # index of maximum iou for each ground true bounding box in each COLUMN
        gt_argmax_ious = ious.argmax(axis=0)
        # the corresponding iou of that maximum index of ground true bounding box
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # the anchor index (the row index) corresponding to the maximum iou of the ground true bounding box
        # in other word, this anchor is the most similar to a particular ground true bounding box
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        # argmax_ious: index of the max iou for each anchor
        # max_ious: the corresponding iou for that max index
        # gt_argmax_ious: the anchor index (the row index) which is most similar to a particular bounding box
        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, W, H):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    xp = cuda.get_array_module(anchor)
    new_rois = xp.zeros((anchor.shape[0], anchor.shape[1] - 1))
    x_coor = xp.vstack((xp.floor(
        anchor[:, 0] - (anchor[:, 2] / 2) * xp.cos(anchor[:, 4]) - (anchor[:, 3] / 2) * xp.sin(anchor[:, 4])),
                             xp.floor(anchor[:, 0] + (anchor[:, 2] / 2) * xp.cos(anchor[:, 4]) - (
                                 anchor[:, 3] / 2) * xp.sin(anchor[:, 4])),
                             xp.floor(anchor[:, 0] - (anchor[:, 2] / 2) * xp.cos(anchor[:, 4]) + (
                                 anchor[:, 3] / 2) * xp.sin(anchor[:, 4])),
                             xp.floor(
                                 anchor[:, 0] + (anchor[:, 2] / 2) * xp.cos(anchor[:, 4]) + (
                                     anchor[:, 3] / 2) * xp.sin(anchor[:, 4]))))
    y_coor = xp.vstack((xp.floor(
        anchor[:, 1] - (anchor[:, 3] / 2) * xp.cos(anchor[:, 4]) + (anchor[:, 2] / 2) * xp.sin(anchor[:, 4])),
                             xp.floor(
                                 anchor[:, 1] - (anchor[:, 3] / 2) * xp.cos(anchor[:, 4]) - (
                                     anchor[:, 2] / 2) * xp.sin(anchor[:, 4])), xp.floor(
        anchor[:, 1] + (anchor[:, 3] / 2) * xp.cos(anchor[:, 4]) + (anchor[:, 2] / 2) * xp.sin(anchor[:, 4])),
                             xp.floor(
                                 anchor[:, 1] + (anchor[:, 3] / 2) * xp.cos(anchor[:, 4]) - (
                                     anchor[:, 2] / 2) * xp.sin(anchor[:, 4]))))
    new_rois[:, 0] = xp.min(x_coor, axis=0)
    new_rois[:, 1] = xp.min(y_coor, axis=0)
    new_rois[:, 2] = xp.max(x_coor, axis=0)
    new_rois[:, 3] = xp.max(y_coor, axis=0)


    index_inside = xp.where(
        (new_rois[:, 0] >= 0) &
        (new_rois[:, 1] >= 0) &
        (new_rois[:, 2] <= W) &  # width
        (new_rois[:, 3] <= H)  # height
    )[0]
    return index_inside


if __name__ == '__main__':
    test = np.array([[40,40,60,60,0],[20,20,5,30,np.pi/4]])
    u = _get_inside_index(test,300,500)
    print u
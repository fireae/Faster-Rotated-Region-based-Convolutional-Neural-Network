from chainer import cuda
import numpy as np
from modules.intersection import intersection_over_union
import math


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 5 or bbox_b.shape[1] != 5:
        raise IndexError

    xp = cuda.get_array_module(bbox_a)
    area = intersection_over_union(bbox_a, bbox_b, xp)
    iou = area / (
        xp.expand_dims(bbox_a[:, 2] * bbox_a[:, 3], axis=1) + xp.expand_dims(bbox_b[:, 2] * bbox_b[:, 3],
                                                                             axis=0) - area)
    iou[xp.isnan(iou)] = 0
    return iou


if __name__ == '__main__':
    bbox1 = np.array([[20, 20, 8, 8, 0], [18, 16, 4, 8, math.pi / 3]])
    bbox2 = np.array([[24, 24, 8, 8, 0], [16, 16, 8, 8, math.pi / 4], [22, 22, 4, 8, math.pi / 6], [24, 24, 8, 8, 0]])

    c = bbox_iou(bbox1, bbox2)
    print c

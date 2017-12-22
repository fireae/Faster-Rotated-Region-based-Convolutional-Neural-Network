from chainer import cuda
import math


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is :math:`p_x, p_y` and
    size :math:`p_w, p_h` and the target bounding box whose center is
    :math:`g_x, g_y` and size :math:`g_w, g_h`, the offsets and scales
    :math:`t_x, t_y, t_w, t_h` can be computed by the following formulas.

    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`

    The output is same type as the type of the inputs.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are used to compute :math:`p_x, p_y, p_w, p_h`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are used to compute :math:`g_x, g_y, g_w, g_h`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    """
    xp = cuda.get_array_module(src_bbox)

    width = src_bbox[:, 2]
    height = src_bbox[:, 3]
    ctr_x = src_bbox[:, 0]
    ctr_y = src_bbox[:, 1]
    angle = src_bbox[:,4]

    base_width = dst_bbox[:, 2]
    base_height = dst_bbox[:, 3]
    base_ctr_x = dst_bbox[:, 0]
    base_ctr_y = dst_bbox[:, 1]
    base_angle = dst_bbox[:,4]

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = xp.log(base_width / width)
    dh = xp.log(base_height / height)
    da = (base_angle - angle)%math.pi

    loc = xp.vstack((dx, dy, dw, dh, da)).transpose()
    return loc

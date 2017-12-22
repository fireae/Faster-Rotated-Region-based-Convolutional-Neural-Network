from chainer import cuda
import math


def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_x, t_y, t_w, t_h` and a bounding
    box whose center is :math:`p_x, p_y` and size :math:`p_w, p_h`,
    the decoded bounding box's center :math:`\\hat{g}_x`, :math:`\\hat{g}_y`
    and size :math:`\\hat{g}_w`, :math:`\\hat{g}_h` are calculated
    by the following formulas.

    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are used to
            compute :math:`p_x, p_y, p_w, p_h`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_x, t_y, t_w, t_h`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_x, \\hat{g}_y, \\hat{g}_w, \\hat{g}_h`.

    """
    xp = cuda.get_array_module(src_bbox)

    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 5), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_width = src_bbox[:, 2]
    src_height = src_bbox[:, 3]
    src_ctr_x = src_bbox[:, 0]
    src_ctr_y = src_bbox[:, 1]
    src_angle = src_bbox[:, 4]

    dx = loc[:, 0::5]
    dy = loc[:, 1::5]
    dw = loc[:, 2::5]
    dh = loc[:, 3::5]
    da = loc[:, 4::5]
    # ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    # ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    # w = xp.exp(dw) * src_width[:, xp.newaxis]
    # h = xp.exp(dh) * src_height[:, xp.newaxis]
    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::5] = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    dst_bbox[:, 1::5] = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    dst_bbox[:, 2::5] = xp.exp(dw) * src_width[:, xp.newaxis]
    dst_bbox[:, 3::5] = xp.exp(dh) * src_height[:, xp.newaxis]
    dst_bbox[:, 4::5] = (da + src_angle[:, xp.newaxis])%math.pi

    if xp.isnan(dst_bbox).any():
        print "there is NAN value"
        raw_input("Press Enter to continue...")
    if xp.isinf(dst_bbox).any():
        print "there is INF value"
        raw_input("Press Enter to continue...")


    return dst_bbox

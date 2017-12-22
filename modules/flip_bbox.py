import math
def flip_bbox(bbox, size, x_flip=False, y_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where the four attributes are coordinates of the bottom left and the
    top right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 5)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The width and the height
            of the image before resized.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    W, H = size
    new_bbox = bbox.copy()
    if x_flip:
        new_bbox[:, 0] = W - 1 - bbox[:, 0]
        new_bbox[:, 1] = bbox[:, 1]
        new_bbox[:, 2] = bbox[:, 3]
        new_bbox[:, 3] = bbox[:, 2]
        new_bbox[:, 4] = math.pi / 2 - bbox[:, 4]
    if y_flip:
        new_bbox[:, 0] = bbox[:, 0]
        new_bbox[:, 1] = H - 1 - bbox[:, 1]
        new_bbox[:, 2] = bbox[:, 3]
        new_bbox[:, 3] = bbox[:, 2]
        new_bbox[:, 4] = math.pi / 2 - bbox[:, 4]
    return new_bbox

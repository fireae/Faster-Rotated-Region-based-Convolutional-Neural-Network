import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset, xp):
    start = int(xp.floor(size * stride))
    end = int(xp.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start


class ROIPooling2D(function.Function):
    """RoI pooling over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, roi_type = in_types
        xp = cuda.get_array_module(roi_type)
        type_check.expect(
            x_type.dtype == xp.float32,
            x_type.ndim == 4,
            roi_type.dtype == xp.float64,
            roi_type.ndim == 2,
            roi_type.shape[1] == 6,
        )

    def forward(self, inputs):
        # self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape
        bottom_data, bottom_rois = inputs
        xp = cuda.get_array_module(bottom_rois)
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        # `xp.zeros` needs to be used because the arrays can be
        # returned without having some of its values updated.
        top_data = xp.zeros((n_rois, channels, self.outh, self.outw),
                            dtype=xp.float32)
        self.argmax_data = xp.zeros(top_data.shape, xp.int32)

        for i_roi in six.moves.range(n_rois):
            # print bottom_rois[i_roi]
            idx, x_center, y_center, roi_width, roi_height, roi_angle = bottom_rois[i_roi]
            xmin = xp.floor(x_center - roi_width / 2)
            xmax = xp.ceil(x_center + roi_width / 2)
            ymin = xp.floor(y_center - roi_height / 2)
            ymax = xp.ceil(y_center + roi_height / 2)

            # idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            xmin = int(max(round(xmin * self.spatial_scale), 0))
            xmax = int(min(round(xmax * self.spatial_scale), width))
            ymin = int(max(round(ymin * self.spatial_scale), 0))
            ymax = int(min(round(ymax * self.spatial_scale), height))

            # print xmin, xmax, ymin, ymax, self.spatial_scale
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            x_slice = slice(xmin, xmax)
            y_slice = slice(ymin, ymax)
            origin_x_coor = xp.arange(width)
            origin_x_coor = xp.tile(origin_x_coor, (height, 1))[y_slice, x_slice].flatten()
            origin_y_coor = xp.arange(height)
            origin_y_coor = xp.tile(origin_y_coor, (width, 1)).transpose()[y_slice, x_slice].flatten()

            transposed_x_coor = xp.clip(((origin_x_coor - x_center) * xp.cos(roi_angle) + (
                origin_y_coor - y_center) * xp.sin(roi_angle) + x_center).astype(xp.int16), 0, width - 1)
            transposed_y_coor = xp.clip((-(origin_x_coor - x_center) * xp.sin(roi_angle) + (
                origin_y_coor - y_center) * xp.cos(roi_angle) + y_center).astype(xp.int16), 0, height - 1)

            out_image = xp.zeros(bottom_data.shape)
            out_image[:, :, y_slice, x_slice] = bottom_data[:, :, transposed_y_coor.tolist(),
                                                transposed_x_coor.tolist()].reshape(
                (bottom_data.shape[0], bottom_data.shape[1], roi_height - 1, roi_width - 1))

            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outh in six.moves.range(self.outh):
                sliceh, lenh = _roi_pooling_slice(
                    outh, strideh, height, ymin, xp)
                if sliceh.stop <= sliceh.start:
                    continue
                for outw in six.moves.range(self.outw):
                    slicew, lenw = _roi_pooling_slice(
                        outw, stridew, width, xmin, xp)
                    if slicew.stop <= slicew.start:
                        continue
                    roi_data = out_image[int(idx), :, sliceh, slicew] \
                        .reshape(channels, -1)
                    top_data[i_roi, :, outh, outw] = \
                        xp.max(roi_data, axis=1)

                    # get the max idx respect to feature_maps coordinates
                    roi_data = cuda.to_cpu(roi_data)
                    max_idx_slice = numpy.unravel_index(
                        numpy.argmax(roi_data, axis=1), (lenh, lenw))  # change back to cpu
                    max_idx_slice = cuda.to_gpu(max_idx_slice)
                    max_idx_slice_h = max_idx_slice[0] + sliceh.start
                    max_idx_slice_w = max_idx_slice[1] + slicew.start
                    max_idx_slice = max_idx_slice_h * width + max_idx_slice_w
                    self.argmax_data[i_roi, :, outh, outw] = max_idx_slice
        print "forward"
        print top_data
        print top_data.shape
        raw_input("Press Enter to continue...")
        return top_data,

    def backward(self, inputs, gy):
        print "begin backward"
        bottom_rois = inputs[1]
        xp = cuda.get_array_module(bottom_rois)
        channels, height, width = self._bottom_data_shape[1:]
        n_rois = bottom_rois.shape[0]
        bottom_delta = xp.zeros(self._bottom_data_shape, xp.float32)

        for i_roi in six.moves.range(n_rois):
            idx, x_center, y_center, roi_width, roi_height, roi_angle = bottom_rois[i_roi]
            xmin = xp.floor(x_center - roi_width / 2)
            xmax = xp.ceil(x_center + roi_width / 2)
            ymin = xp.floor(y_center - roi_height / 2)
            ymax = xp.ceil(y_center + roi_height / 2)
            # idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            idx = int(idx)
            xmin = int(max(round(xmin * self.spatial_scale), 0))
            xmax = int(min(round(xmax * self.spatial_scale), width))
            ymin = int(max(round(ymin * self.spatial_scale), 0))
            ymax = int(min(round(ymax * self.spatial_scale), height))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)

            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)

            # iterate all the w, h (from feature map) that fall into this ROIs
            for w in six.moves.range(xmin, xmax + 1):
                for h in six.moves.range(ymin, ymax + 1):
                    phstart = int(xp.floor(float(h - ymin) / strideh))
                    phend = int(xp.ceil(float(h - ymin + 1) / strideh))
                    pwstart = int(xp.floor(float(w - xmin) / stridew))
                    pwend = int(xp.ceil(float(w - xmin + 1) / stridew))

                    phstart = min(max(phstart, 0), self.outh)
                    phend = min(max(phend, 0), self.outh)
                    pwstart = min(max(pwstart, 0), self.outw)
                    pwend = min(max(pwend, 0), self.outw)

                    for ph in six.moves.range(phstart, phend):
                        for pw in six.moves.range(pwstart, pwend):
                            max_idx_tmp = self.argmax_data[i_roi, :, ph, pw]
                            for c in six.moves.range(channels):
                                if max_idx_tmp[c] == (h * width + w):
                                    bottom_delta[idx, c, h, w] += \
                                        gy[0][i_roi, c, ph, pw]
        print "backward"
        print bottom_delta
        raw_input("Press Enter to continue...")
        return bottom_delta, None


def roi_pooling_2d(x, rois, outh, outw, spatial_scale):
    """Spatial Region of Interest (ROI) pooling function.

    This function acts similarly to :class:`~functions.MaxPooling2D`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 5), and each datum is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        outh (int): Height of output image after pooled.
        outw (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return ROIPooling2D(outh, outw, spatial_scale)(x, rois)

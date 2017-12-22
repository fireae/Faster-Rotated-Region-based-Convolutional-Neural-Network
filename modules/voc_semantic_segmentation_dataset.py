import numpy as np
import os
from math import sin, cos, radians, floor

import chainer
import cv2
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCSemanticClassSegmentationDataset(chainer.dataset.DatasetMixin):
    """Dataset class for the semantic segmantion task of PASCAL `VOC2012`_.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_semantic_segmentation_label_names`.


    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'trainval', 'val', 'test']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2007', split)
            # data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. The color image is in CHW
        format and the label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the
            images. The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img_file = os.path.join(
            self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_file, color=True)
        label = self._load_label(self.data_dir, self.ids[i])
        return img, label

    def _load_label(self, data_dir, id_):
        label_file = os.path.join(
            data_dir, 'SegmentationClass', id_ + '.png')
        label = read_image(label_file, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]


class VOCSemanticObjectSegmentationDataset(chainer.dataset.DatasetMixin):
    """Dataset class for the semantic segmantion task of PASCAL `VOC2012`_.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_semantic_segmentation_label_names`.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'trainval', 'val', 'test']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2007', split)
            # data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. The color image is in CHW
        format and the label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the
            images. The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img_file = os.path.join(
            self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_file, color=True)
        label = self._load_label(self.data_dir, self.ids[i])
        return img, label

    def _load_label(self, data_dir, id_):
        label_file = os.path.join(
            data_dir, 'SegmentationObject', id_ + '.png')
        label = read_image(label_file, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]

class VOCSemanticSegmentationDataset(chainer.dataset.DatasetMixin):
    """Dataset class for the semantic segmantion task of PASCAL `VOC2012`_.



    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'trainval', 'val', 'test']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2007', split)
            # data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.split = split
        self.data_dir = data_dir
        self.dataset_class = VOCSemanticClassSegmentationDataset(split=self.split)
        self.dataset_object = VOCSemanticObjectSegmentationDataset(split=self.split)

    def __len__(self):
        return len(self.ids)

    def get_example(self, index):
        """Returns the i-th example.

        Returns a color image and a label image. The color image is in CHW
        format and the label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the
            images. The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if index >= len(self):
            raise IndexError('index is too large')
        bboxs = []
        labels = []
        # dataset_class = VOCSemanticClassSegmentationDataset(split=self.split)
        # print len(dataset_class)
        img, label_class = self.dataset_class[index]
        original_img = np.copy(img)
        # dataset_object = VOCSemanticObjectSegmentationDataset(split=self.split)
        # print len(dataset_object)
        _, label_object = self.dataset_object[index]
        # print np.max(label_object)
        img = img.transpose((1, 2, 0))
        for i in range(1, np.max(label_object) + 1):
            grayscale_image = np.copy(label_object)

            grayscale_image[grayscale_image != i] = 0
            grayscale_image[grayscale_image == i] = 255

            grayscale_image = grayscale_image.astype(np.uint8)
            # cv2.imwrite("grayscale.png", grayscale_image)

            ret, thresh = cv2.threshold(grayscale_image, 127, 255, 0)
            _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

            LENGTH = len(contours)
            status = np.zeros((LENGTH, 1))

            for i, cnt1 in enumerate(contours):
                x = i
                if i != LENGTH - 1:
                    for j, cnt2 in enumerate(contours[i + 1:]):
                        x = x + 1
                        dist = find_if_close(cnt1, cnt2)
                        if dist == True:
                            val = min(status[i], status[x])
                            status[x] = status[i] = val
                        else:
                            if status[x] == status[i]:
                                status[x] = i + 1

            unified = []
            maximum = int(status.max()) + 1
            for i in xrange(maximum):
                pos = np.where(status == i)[0]
                if pos.size != 0:
                    cont = np.vstack(contours[i] for i in pos)
                    hull = cv2.convexHull(cont)
                    unified.append(hull)
            # cv2.drawContours(img, unified, -1, (0, 255, 0), 2)
            cv2.drawContours(thresh, unified, -1, 255, -1)

            _, i_contours, i_hierarchy = cv2.findContours(thresh, 1, 2)
            i_cnt = i_contours[0]
            rect = draw_oriented_box(img, i_cnt)
            bboxs.append(
                [floor(rect[0][0]), floor(rect[0][1]), floor(rect[1][0]), floor(rect[1][1]), radians(-rect[2])])
            labels.append(label_class[np.argwhere(grayscale_image > 0)[0][0], np.argwhere(grayscale_image > 0)[0][1]])

        bboxs = np.asarray(bboxs).astype(np.float64)

        #Begin Not Important
        # cv2.imwrite("original_color.png", img)
        #End Not Important

        return original_img, bboxs, labels

    def _load_label(self, data_dir, id_):
        label_file = os.path.join(
            data_dir, 'SegmentationClass', id_ + '.png')
        label = read_image(label_file, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]


def draw_oriented_box(img, cnt):
    rect = cv2.minAreaRect(cnt)
    # print rect
    standalized_rect = ()
    standalized_rect += (rect[0],)
    if rect[2] > -45:
        standalized_rect += ((rect[1][0], rect[1][1]),)
        standalized_rect += (rect[2],)
    else:
        standalized_rect += ((rect[1][1], rect[1][0]),)
        standalized_rect += (90+rect[2],)

    # Begin Not Important
    # box = cv2.boxPoints(standalized_rect)
    # box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (255, 255, 255), 2)
    # End Not Important

    return standalized_rect


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]

    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 500:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False

def roi_pooling(bottom_data,bbox,xp):
    height, width = bottom_data.shape[2:]
    x_slice = slice(max(xp.floor(bbox[1] - bbox[3] / 2).astype(xp.int16),0), min(xp.ceil(bbox[1] + bbox[3] / 2).astype(xp.int16),width))
    y_slice = slice(max(xp.floor(bbox[2] - bbox[4] / 2).astype(xp.int16),0), min(xp.ceil(bbox[2] + bbox[4] / 2).astype(xp.int16),height))

    origin_x_coor = xp.arange(width)
    origin_x_coor = xp.tile(origin_x_coor,(height,1))[y_slice,x_slice].flatten()
    origin_y_coor = xp.arange(height)
    origin_y_coor = xp.tile(origin_y_coor,(width,1)).transpose()[y_slice,x_slice].flatten()

    transposed_x_coor = xp.clip(((origin_x_coor - bbox[1]) * xp.cos(bbox[5]) + (origin_y_coor - bbox[2]) * xp.sin(bbox[5]) + bbox[1]).astype(np.int),0,width-1)
    transposed_y_coor = xp.clip((-(origin_x_coor - bbox[1]) * xp.sin(bbox[5]) + (origin_y_coor - bbox[2]) * xp.cos(bbox[5]) + bbox[2]).astype(np.int),0,height-1)

    out_image = xp.zeros(bottom_data.shape)
    out_image[:,:,origin_y_coor.tolist(),origin_x_coor.tolist()] = bottom_data[:,:,transposed_y_coor.tolist(),transposed_x_coor.tolist()]

    return out_image

if __name__ == '__main__':
    dataset = VOCSemanticSegmentationDataset(split="trainval")

    img, bbox, label = dataset[119]
    print(img.shape)
    print(bbox)
    print(label)
    # for i in range(len(dataset)):
    #     print i
    #     img, bbox, label = dataset[i]
    #     for j in range(len(bbox)):
    #         out_img = roi_pooling(np.expand_dims(img, axis=0), np.insert(bbox[j], 0, 0), np)
    #         out_img = out_img.reshape((out_img.shape[1], out_img.shape[2], out_img.shape[3])).transpose((1, 2, 0))
    #         cv2.imwrite("standard_output/tranposed_color_{}_{}.png".format(i, j), out_img)

    print('len:', len(dataset))

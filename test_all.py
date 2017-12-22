import argparse
import matplotlib.pyplot as plot

import chainer

from modules.voc_utils import voc_detection_label_names
from modules.faster_rcnn_vgg import FasterRCNNVGG16
import cv2 as cv
import numpy as np
from math import sin, cos, radians, floor, pi
from modules.voc_semantic_segmentation_dataset import VOCSemanticSegmentationDataset

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def draw_result(out, bbox, clss):
    CV_AA = 16
    out = out.transpose((1, 2, 0))
    print bbox
    print bbox.shape
    print clss
    for row in range(len(bbox)):
        center_x, center_y, weight, height, angle = map(int,bbox[row])
        if angle%(2*pi) > pi/4:
            t = weight
            weight = height
            height = t
            # angle = pi/2 - angle
        point_1 = [floor(center_x - (weight / 2) * cos(angle) - (height / 2) * sin(angle)), floor(
            center_y - (height / 2) * cos(angle) + (weight / 2) * sin(angle))]
        point_2 = [floor(center_x + (weight / 2) * cos(angle) - (height / 2) * sin(angle)), floor(
            center_y - (height / 2) * cos(angle) - (weight / 2) * sin(angle))]
        point_3 = [floor(center_x + (weight / 2) * cos(angle) + (height / 2) * sin(angle)), floor(
            center_y + (height / 2) * cos(angle) - (weight / 2) * sin(angle))]
        point_4 = [floor(center_x - (weight / 2) * cos(angle) + (height / 2) * sin(angle)), floor(
            center_y + (height / 2) * cos(angle) + (weight / 2) * sin(angle))]

        points = [point_1, point_2, point_3, point_4]
        points = np.int0(points)

        cv.drawContours(out, [points], 0, (255,0,255), 2, CV_AA)
        # ret, baseline = cv.getTextSize(
        #     CLASSES[clss[row]+1], cv.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        # cv.rectangle(out, (x1, y2 - ret[1] - baseline),
        #              (x1 + ret[0], y2), (0, 0, 255), -1)
        # cv.putText(out, CLASSES[clss[row]+1], (x1, y2 - baseline),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, CV_AA)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained_model', default='pretrained_model/snapshot_model.npz')
    # parser.add_argument('image')
    parser.add_argument('--out_fn', type=str, default='result.jpg')
    args = parser.parse_args()

    model = FasterRCNNVGG16(
        n_fg_class=len(voc_detection_label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()

    # img = utils.read_image(args.image, color=True)
    dataset = VOCSemanticSegmentationDataset(split="trainval")
    for i in range(len(dataset)):
        print i
        img, _, _ = dataset[i]
        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        result = draw_result(img, bbox, label)
        cv.imwrite("predict_all/image_{}.jpg".format(i), result)


if __name__ == '__main__':
    main()

# Faster-Rotated-Region-based-Convolutional-Neural-Network
The state-of-the-art object detection networks for natural images have recently demonstrated impressive performances. However the complexity of object's shape and rotation expose the limited capacity of these networks for strip-like rotated assembled object detection which are common 
in any dataset as well as real images. In this project, I embrace this observation and introduce the Faster Rotated Region-based Convolutional Neural Network Faster RR-CNN, 
which can learn and accurately extract features of rotated regions and locate rotated objects precisely. In comparison with the classic Faster RCNN, Faster RR-CNN has three important new components including a skew non-maximum suppression, a rotated bounding box regression model and  a rotated region of interest (RRoI) pooling layer. I conduct experiment using the VOC2012 dataset, demonstrating the potential ability of this novel network in detecting oriented objects.

In order to train the model with either VOC2007 or VOC2012 dataset, you could run:

  python train.py -m VOC2xxx

In order to test model with new image, you could run:

  python test.py -p path/to/the/image

#!/usr/bin/env python3
"""Yolo Module - Based on 0-yolo.py"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Class uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """

        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for
        the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the
        initial filtering step
        nms_t is a float representing the IOU threshold for non-max
        suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
        outputs is the number of outputs (predictions) made by the Darknet
        model
        anchor_boxes is the number of anchor boxes used for each prediction
        2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes

        """
        # load the model
        self.model = K.models.load_model(model_path)

        # load the class labels
        with open(classes_path, 'rt') as file:
            self.class_names = file.read().rstrip('\n').split('\n')

        # setting the thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """

        outputs is a list of numpy.ndarrays containing
        the predictions from the Darknet model for a single image:
        Each output will have the shape (grid_height, grid_width, anchor_boxes, 
        4 + 1 + classes)
        grid_height & grid_width => the height and width of the grid used for
        the output
        anchor_boxes => the number of anchor boxes used
        4 => (t_x, t_y, t_w, t_h)
        1 => box_confidence
        classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image's original size
        [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes,
        4) containing the processed boundary boxes for each output, respectively:
        4 => (x1, y1, x2, y2)
        (x1, y1, x2, y2) should represent the boundary box relative to original image
        box_confidences: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, 1) containing the box confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, classes) containing the box's class probabilities for each output, respectively

        """
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w = output.shape[:2]
            anchor_boxes = self.anchors[i].shape[0]

            txy = output[..., :2]
            twh = output[..., 2:4]
            conf_sigmoid = self.sigmoid(output[..., 4:5])
            prob_sigmoid = self.sigmoid(output[..., 5:])

            conf_box = np.expand_dims(conf_sigmoid, axis=-1)

            box_confidences.append(conf_box)
            box_class_probs.append(prob_sigmoid)

            box_wh = self.anchors[i] * np.exp(twh)
            box_wh /= [grid_w, grid_h]

            grid = np.tile(np.indices((grid_w, grid_h)).T, anchor_boxes).reshape(grid_h, grid_w, -1, 2)

            box_xy = (self.sigmoid(txy) + grid) / [grid_w, grid_h]
            box_xy1 = box_xy - (box_wh / 2)
            box_xy2 = box_xy + (box_wh / 2)

            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            box *= np.tile(image_size, 2)

            boxes.append(box)

        # returns a tuple of each
        return (boxes, box_confidences, box_class_probs)

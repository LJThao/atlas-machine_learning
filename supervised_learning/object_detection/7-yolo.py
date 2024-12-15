#!/usr/bin/env python3
"""Yolo Module - Based on 6-yolo.py"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


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
        Each output will have the shape (grid_height, grid_width, anchor_boxes
        , 4 + 1 + classes)
        grid_height & grid_width => the height and width of the grid used for
        the output
        anchor_boxes => the number of anchor boxes used
        4 => (t_x, t_y, t_w, t_h)
        1 => box_confidence
        classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image's original size
        [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes,
        4) containing the processed boundary boxes for each output,
        respectively:
        4 => (x1, y1, x2, y2)
        (x1, y1, x2, y2) should represent the boundary box relative to
        original image
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 1) containing the box confidences for
        each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, classes) containing the box's class
        probabilities for each output, respectively

        """
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]
            anchors = self.anchors[i]

            txy = output[..., :2]
            twh = output[..., 2:4]
            conf_sigmoid = self.sigmoid(output[..., 4:5])
            prob_sigmoid = self.sigmoid(output[..., 5:])

            conf_box = np.expand_dims(conf_sigmoid, axis=-1)
            box_confidences.append(conf_box)
            box_class_probs.append(prob_sigmoid)

            box_wh = anchors * np.exp(twh)
            box_wh /= [grid_width, grid_height]

            grid = np.indices((grid_width, grid_height)).T.reshape(
                grid_height,
                grid_width,
                1, 2)

            grid = np.tile(grid, (1, 1, anchors.shape[0], 1))

            box_xy = (self.sigmoid(txy) + grid) / [grid_width, grid_height]

            box_xy1 = box_xy - (box_wh / 2)
            box_xy2 = box_xy + (box_wh / 2)

            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            box *= np.tile(image_size, 2)

            boxes.append(box)

        # returns a tuple of each
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """

        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, 4) containing the processed boundary boxes for each
        output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 1) containing the processed box confidences
        for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, classes) containing the processed box class
        probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        that each box in filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively

        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_confidence = box_confidences[i].squeeze(axis=-1)
            box_class_prob = box_class_probs[i]
            box_score = box_confidence * box_class_prob
            box_class = np.argmax(box_score, axis=-1)
            box_score = np.max(box_score, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """

        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
        the filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores):
        box_predictions: a numpy.ndarray of shape (?, 4) containing all of the
        predicted bounding boxes ordered by class and box score
        predicted_box_classes: a numpy.ndarray of shape (?,) containing the
        class number for box_predictions ordered by class and box score,
        respectively
        predicted_box_scores: a numpy.ndarray of shape (?) containing the box
        scores for box_predictions ordered by class and box score, respectively

        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in set(box_classes):
            idx = np.where(box_classes == c)

            filtered_boxes_c = filtered_boxes[idx]
            box_scores_c = box_scores[idx]
            box_classes_c = box_classes[idx]

            pick = self._nms(filtered_boxes_c, box_scores_c)

            box_predictions.append(filtered_boxes_c[pick])
            predicted_box_classes.append(box_classes_c[pick])
            predicted_box_scores.append(box_scores_c[pick])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    def _nms(self, boxes, scores):
        """

        Non-Maximum Suppression to filter out overlapping boxes.

        """
        pick = []
        order = scores.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            pick.append(i)
            iou = self._iou(boxes[i], boxes[order[1:]])
            order = order[1:][iou < self.nms_t]

        return (np.array(pick))

    def _iou(self, box1, boxes):
        """

        Compute Intersection over Union (IoU).

        """
        x1, y1, x2, y2 = box1
        x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:,
                                                                          3]

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(
            0, inter_y2 - inter_y1)

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2s - x1s) * (y2s - y1s)

        union_area = area1 + area2 - inter_area

        return (inter_area / union_area)

    @staticmethod
    def load_images(folder_path):
        """

        folder_path: a string representing the path to the folder
        holding all the images to load
        Returns a tuple of (images, image_paths):
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images

        """
        images, image_paths = [], []

        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                image_paths.append(path)

        return (images, image_paths)

    def preprocess_images(self, images):
        """

        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
        pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3) containing
        all of the preprocessed images
        ni: the number of images that were preprocessed
        input_h: the input height for the Darknet model Note: this can vary by
        model
        input_w: the input width for the Darknet model Note: this can vary by
        model
        3: number of color channels
        image_shapes: a numpy.ndarray of shape (ni, 2) containing the original
        height and width of the images
        2 => (image_height, image_width)

        """
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        image_shapes = [image.shape[:2] for image in images]
        pimages = []

        for image in images:
            resized_image = cv2.resize(image,
                                       (input_w, input_h),
                                       interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image / 255.0
            pimages.append(resized_image)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)
    
    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """

        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored

        """
        for box, box_class_idx, box_score in zip(boxes,
                                                 box_classes,
                                                 box_scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.class_names[box_class_idx]} {box_score:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1)

        cv2.imshow(file_name, image)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            os.makedirs('./detections/', exist_ok=True)
            cv2.imwrite(f'./detections/{file_name}', image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """

        folder_path: a string representing the path to the folder holding all
        the images to predict
        All image windows should be named after the corresponding image
        filename without its full path(see examples below)
        Displays all images using the show_boxes method
        Returns: a tuple of (predictions, image_paths):
        predictions: a list of tuples for each image of (boxes, box_classes,
        box_scores)
        image_paths: a list of image paths corresponding to each prediction in
        predictions

        """
        images, image_paths = self.load_images(folder_path)
        preprocessed_images, image_shapes = self.preprocess_images(images)

        pre_image = np.stack(preprocessed_images, axis=0)
        raw_predictions = self.model.predict(pre_image)

        predictions = []

        for i, image in enumerate(images):
            one_raw = [raw_predictions[j][i] for j in range(3)]
            boxes, confidences, class_probs = self.process_outputs(one_raw,
                                                                   image_shapes[i])
            filtered_boxes, box_classes, box_scores = self.filter_boxes(boxes,
                                                                        confidences,
                                                                        class_probs)
            box_preds, class_preds, score_preds = self.non_max_suppression(filtered_boxes,
                                                                           box_classes,
                                                                           box_scores)

            self.show_boxes(image, box_preds, class_preds,
                            score_preds, image_paths[i].split('/')[-1])

            predictions.append((box_preds, class_preds, score_preds))

        return (predictions, image_paths)

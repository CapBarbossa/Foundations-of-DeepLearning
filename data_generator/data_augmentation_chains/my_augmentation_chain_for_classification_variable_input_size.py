from data_generator.transformations.object_detection_2d_image_boxes_validation_utils import BoxFilter
from .data_augmentation_chain_base import *
from data_generator.transformations.object_detection_2d_geometric_ops import RandomFlip_arbitrary, ResizeRandomInterp


class DataAugmentationVariableInputSize(object):
    '''
    Applies a chain of photometric and geometric image transformations. For documentation, please refer
    to the documentation of the individual transformations involved.

    Important: This augmentation chain is suitable for variable-size images only.
    '''

    def __init__(self,
                 resize_height,
                 resize_width,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''

        :param resize_height:
        :param resize_width:
        :param background:
        :param labels_format: 这个参数只是为了标记label字段
        '''
        self.photometric_distortions = SSDPhotometricDistortions()
        self.expand = SSDExpand()
        self.background = background
        self.labels_format = labels_format

        # Utility transformations
        self.resize = ResizeRandomInterp(height=resize_height,
                                                   width=resize_width,
                                                   interpolation_modes=[cv2.INTER_NEAREST,
                                                                        cv2.INTER_LINEAR,
                                                                        cv2.INTER_CUBIC,
                                                                        cv2.INTER_AREA,
                                                                        cv2.INTER_LANCZOS4],
                                                   box_filter=None,
                                                   labels_format=self.labels_format)

        # Geometric transformations
        # self.random_flip = RandomFlip_arbitrary(dim='horizontal', prob=0.5, labels_format=self.labels_format)
        # Add swapping channels
        self.random_channel_swap = RandomChannelSwap(prob=0.5)

        # Define the processing chain
        self.transformations = [
            self.photometric_distortions,
            self.expand,
            self.random_channel_swap,
            self.resize,
        ]

    def __call__(self, image, labels=None):

        # self.random_flip.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format

        if not (labels is None):
            for transform in self.transformations:
                image, labels = transform(image, labels)
            return image, labels
        else:
            for transform in self.transformations:
                image = transform(image)
            return image

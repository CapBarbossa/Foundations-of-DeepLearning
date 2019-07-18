from __future__ import division
import keras


class InputEncoder(object):
    def __init__(self,
                 num_class,
                 ):
        self.num_classes = num_class

    def __call__(self, ground_truth_labels, diagnostics=False):
        return keras.utils.to_categorical(ground_truth_labels, self.num_classes)















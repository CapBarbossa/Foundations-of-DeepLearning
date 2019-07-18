from __future__ import division
import numpy as np


class InputEncoder(object):
    def __init__(self,
                 NetInput_height,
                 NetInput_width,
                 variance=[0.1, 0.1],
                 ):
        self.img_height = NetInput_height
        self.img_width  = NetInput_width
        self.variance   = variance

    def __call__(self, ground_truth_labels, diagnostics=False):
        x1, y1 = 0, 1
        batch_size = len(ground_truth_labels)
        y_encoded = np.zeros_like(ground_truth_labels, dtype=np.float32)
        for i in range(batch_size):
            if ground_truth_labels[i].size == 0: continue
            labels = ground_truth_labels[i].astype(np.float32)
            labels[:, [y1, ]] /= self.img_height
            labels[:, [y1, ]] /= self.variance[1]
            labels[:, [x1, ]] /= self.img_width
            labels[:, [x1, ]] /= self.variance[0]
            y_encoded[i] = labels

        return np.squeeze(y_encoded, axis=1)    # 解释：这里加上squeeze的原因在于datagenerator直接的输出结果的尺寸为[batchsize, 1, num_of_labels],而不是预期的[batchsize, num_of_labels]















'''
本脚本最下方的
    if __name__ == "__main__": 是为了测试数据变换的正确性，有可视化结果。
    如果对本数据增加的部分有疑问，可以直接找到相应的数据增加函数，调整，然后再到这里运行。这个脚本本身并不实现具体的图像数据变换操作，而只是调用这些变换操作。
本脚本只能够从硬盘中读取数据的方式提供生成器对象，不能够使用任何hdf5文件!!!
'''

from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys, glob
from tqdm import tqdm, trange

try:
    import h5py
except ImportError:
    warnings.warn("'h5py' module is missing. The fast HDF5 dataset option will be unavailable.")
try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn(
        "'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

from data_generator.ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.transformations.object_detection_2d_image_boxes_validation_utils import BoxFilter


class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass


class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass


class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    本脚本的数据生成器面向的是分类任务. 数据组织方式不再是记录文件(label.csv)，而是直接的文件夹---即形如下面的字典对象
                dict({
                        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Notgray', 'index0'): 0,
                        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Notgray', 'index1'): 1,
                        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Notgray', 'index2'): 2,
                        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Isgray', 'index0'): 3,
                        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Isgray', 'index1'): 4,
                        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Isgray', 'index2'): 5,
                        '/home/bro/MyWork/job8_皇室战争/DL/traindata03/NotinGame': 6,
                    })
    常用的分类任务使用这种方式非常直观、易懂，一个单独的文件夹就是一个类别，清晰！不再专门把这些不同位置的图像文件集中起来，然后将他们对应的class ID编写进一个txt或者csv文件中，然后打乱，然后按照一定的比例划分训练和验证集。这些琐碎的细节，全部给我滚到程序内部中去。
    Note: 1.label必须是整数，而不是什么浮点数值 2.shuffle操作的对象是index列表，而非直接的图像或者label所在的数组 3.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.

    load_images_into_memory:是否读取图片数据到python中，这就是载入内存的意思.
    '''

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id',),
                 verbose=True):
        '''
        Initializes the data generator. You can either load a dataset directly here in the constructor,
        e.g. an HDF5 dataset, or you can use one of the parser methods to read in a dataset.

        Arguments:
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            hdf5_dataset_path (str, optional): The full file path of an HDF5 file that contains a dataset in the
                format that the `create_hdf5_dataset()` method produces. If you load such an HDF5 dataset, you
                don't need to use any of the parser methods anymore, the HDF5 dataset already contains all relevant
                data.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
            eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
                a list that indicates for each ground truth object in the image whether that object is supposed
                to be treated as neutral during an evaluation.
            labels_output_format (list, optional): A list of five strings representing the desired order of the
                items class ID, in the generated ground truth data (if any). The expected strings are 'class_id'.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
        '''
        self.labels_output_format = labels_output_format
        self.labels_format = {
            'class_id': labels_output_format.index('class_id'), }  # This dictionary is for internal use.

        self.dataset_size = 0  # As long as we haven't loaded anything yet, the dataset size is zero.
        self.load_images_into_memory = load_images_into_memory
        self.images = None  # The only way that this list will not stay `None` is if `load_images_into_memory == True`.

        # `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves. This list is one of the outputs of the parser methods.
        # In case you are loading an HDF5 dataset, this list will be `None`.
        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError(
                    "`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose:
                    it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else:
                    it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
        # of ground truth bounding boxes for that image.
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError(
                    "`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

        if not hdf5_dataset_path is None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None

    def load_hdf5_dataset(self, verbose=True):
        '''
        Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method
        produces.

        Arguments:
            verbose (bool, optional): If `True`, prints out the progress while loading
                the dataset.

        Returns:
            None.
        '''

        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size,
                                         dtype=np.int32)  # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def get_filenames_labels(self, src_dict: dict, split_ratio):
        '''
        本函数所返回的名称列表的顺序并未被打乱，等到在数据生成器中再进行打乱操作
        :param src_dict: key-各种类训练数据的路径 value-类别数值 int， counting from 0
        :param split_ratio: 这这是由组织训练数据的方式决定的-在硬盘上，存放训练数据的文件夹只有一个训练集，没有验证集，那么验证集只有从训练数据中分离一部分出来，那么就需要一个比例来决定抽取多少作为训练，多少作为验证.同时，这也方便了制作训练数据的步骤，我只需要弄足够多的训练数据即可，将它们放在不同的文件夹中作为不同的类别，而不需要一模一样的操作做两次，专门弄一个验证目录。注意！这也是不常规的一个点，并且是很重要的一个点，毕竟是数据操作，要警惕“从一开始就错了”这种陷阱.
        :return: 二个列表二个数组，一个存放文件名称，一个存放相对应的label数值，第三个数组存放程序自动划分出来的训练数据的索引集合，第四个数组存放验证集合的索引列表.
        '''
        filenames, labels = [], []
        for key, value in src_dict.items():
            cur_fnames = glob.glob(os.path.join(key, '*'))
            cur_labels = [value] * len(cur_fnames)
            filenames += cur_fnames
            labels += cur_labels
        # 这里划分训练和验证集时，随机分开的“随机”是必要的.
        tsize = len(filenames)
        split_spot = int(tsize * split_ratio)
        random_index = np.random.permutation(tsize).astype('int32')
        self.train_indexes = random_index[:split_spot]  # 训练部分对应的索引
        self.val_indexes = random_index[split_spot:]  # 验证部分对应的索引
        self.dataset_size_train = split_spot
        self.dataset_size_val = tsize - split_spot
        self.filenames = np.array(filenames)
        self.labels = np.array(labels)

    def _create_hdf5_dataset(self,
                             data_size, data_indice,
                             file_path='trainset.h5',
                             resize=False,
                             variable_image_size=True,
                             verbose=True):
        '''
        Converts the currently loaded dataset into a HDF5 file. This HDF5 file contains all
        images as uncompressed arrays in a contiguous block of memory, which allows for them
        to be loaded faster. Such an uncompressed dataset, however, may take up considerably
        more space on your hard drive than the sum of the source images in a compressed format
        such as JPG or PNG.

        It is recommended that you always convert the dataset into an HDF5 dataset if you
        have enugh hard drive space since loading from an HDF5 dataset accelerates the data
        generation noticeably.

        Note that you must load a dataset (e.g. via one of the parser methods) before creating
        an HDF5 dataset from it.

        The created HDF5 dataset will remain open upon its creation so that it can be used right
        away.

        Arguments:
            file_path (str, optional): The full file path under which to store the HDF5 dataset.
                You can load this output file via the `DataGenerator` constructor in the future.
            resize (tuple, optional): `False` or a 2-tuple `(height, width)` that represents the
                target size for the images. All images in the dataset will be resized to this
                target size before they will be written to the HDF5 file. If `False`, no resizing
                will be performed.
            variable_image_size (bool, optional): The only purpose of this argument is that its
                value will be stored in the HDF5 dataset in order to be able to quickly find out
                whether the images in the dataset all have the same size or not.
            verbose (bool, optional): Whether or not prit out the progress of the dataset creation.

        Returns:
            None.
        '''

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(file_path, 'w')

        # Create a few attributes that tell us what this dataset contains.
        # The dataset will obviously always contain images, but maybe it will
        # also contain labels, image IDs, etc.
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
        # It's useful to be able to quickly check whether the images in a dataset all
        # have the same size or not, so add a boolean attribute for that.
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # Create the dataset in which the images will be stored as flattened arrays.
        # This allows us, among other things, to store images of variable size.
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(data_size,),
                                                  maxshape=(None),
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # Create the dataset that will hold the image heights, widths and channels that
        # we need in order to reconstruct the images from the flattened arrays later.
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(data_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        if not (self.labels is None):
            # Create the dataset in which the labels will be stored as flattened arrays.
            # the dtype in here indicates that, labels in files or dict should be integer， 不对呀，那回归时候用的小数label算咋说？
            # 回归的时候，label文件中使用的是整数！
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(data_size,),
                                                      maxshape=(None),
                                                      dtype=h5py.special_dtype(vlen=np.int32))

            # Create the dataset that will hold the dimensions of the labels arrays for
            # each image so that we can restore the labels from the flattened arrays later.
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                            shape=(data_size,),
                                                            maxshape=(None,),
                                                            dtype=np.int32)

            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if not (self.image_ids is None):
            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=(None),
                                                         dtype=h5py.special_dtype(vlen=str))

            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

        if not (self.eval_neutral is None):
            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=np.bool_))

            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        if verbose:
            tr = trange(data_size, desc='Creating HDF5 dataset', file=sys.stdout)
        else:
            tr = range(data_size)

        # Iterate over all images in the dataset.
        for i in tr:

            # Store the image.
            with Image.open(self.filenames[data_indice[i]]) as image:  # Trick: 这里使用with语句,直接过滤掉了那些图像数据损坏的文件.

                image = np.asarray(image, dtype=np.uint8)

                # Make sure all images end up having three channels.
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # Flatten the image array and write it to the images dataset.
                hdf5_images[i] = image.reshape(-1)
                # Write the image's shape to the image shapes dataset.
                hdf5_image_shapes[i] = image.shape

            # Store the ground truth if we have any.
            if not (self.labels is None):
                labels = np.asarray(self.labels[data_indice[i]])
                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)
                # Write the labels' shape to the label shapes dataset.
                hdf5_label_shapes[i] = 1,

            # Store the image ID if we have one.
            if not (self.image_ids is None):
                hdf5_image_ids[i] = self.image_ids[i]

            # Store the evaluation-neutrality annotations if we have any.
            if not (self.eval_neutral is None):
                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()
        # 这里的逻辑：既然创建了h5文件，不能仅仅创建完以后扔那，而是马上利用起来，所以，这里需要返回一个h5文件的句柄;其次，这里还返回了新的索引，原因在于...旧的索引他娘的是对应的随机分开的数据，而转化为hdf5_dataset以后，它的顺序变为了顺序，因而。。。他娘的应该是没有讲清楚，下次来的时候在看看～～～
        hdf5_dataset = h5py.File(file_path, 'r')
        return hdf5_dataset, np.arange(data_size, dtype=np.int32)

    def create_hdf5_dataset(self, file_path, if_resize=False):
        '''
        由于不可力抗的因素，创建hdf5的项目搁浅.原因在于 index的随机与不随机的矛盾。 What a SHAME!
        :param file_path:
        :param if_resize: 是否在存入hdf5文件之前，resize输入样本.
        :return:
        '''
        self.hdf5_dataset_train, self.train_indexes = self._create_hdf5_dataset(self.dataset_size_train,
                                                                                self.train_indexes,
                                                                                file_path=file_path[0],
                                                                                resize=if_resize,
                                                                                variable_image_size=True, verbose=True)
        self.hdf5_dataset_val, self.val_indexes = self._create_hdf5_dataset(self.dataset_size_val, self.val_indexes,
                                                                            file_path=file_path[1],
                                                                            resize=if_resize, variable_image_size=True,
                                                                            verbose=True)

    def generate(self, label_encoder,
                 batch_size=[32, 16], shuffle=[True, False],
                 transformations=[[], []]):
        '''
        本函数在生成数据的过程中，只应用数据增广，不应用label_encoder(正常情况下应当是有encoder的)。因为这个函数是面向简单分类任务的，因此，对于通常是0-20以内的整数数字的label，不作任何处理！
        本函数的逻辑是：src_dict包含了所有的训练和验证数据，读取一次以后，在函数内部以split_ratio进行分割训练和验证集，然后在内部生成两个数据生成器，就可以了！不必再分两次创建生成器对象，读取两次目录了。(为了实现这一目标效果，耗费了2019.07.15-18四天时间，真他娘的狠.)
        :param label_encoder:
        :param src_dict:
        :param split_ratio:
        :param batch_size:
        :param shuffle:
        :param transformations:
        :return:
        '''
        # self.get_filenames_labels(src_dict, split_ratio)
        train = self._generate(self.dataset_size_train, self.train_indexes,
                               label_encoder,
                               None, batch_size[0],
                               shuffle[0], transformations[0])
        val = self._generate(self.dataset_size_val, self.val_indexes,
                             label_encoder,
                             None, batch_size[1], shuffle[1],
                             transformations[1])
        return train, val

    def _generate(self,
                  data_size,
                  data_indice,
                  label_encoder,
                  hdf5_dataset=None,
                  batch_size=32,
                  shuffle=True,
                  transformations=[],
                  keep_images_without_gt=False,
                  degenerate_box_handling='remove'):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        函数逻辑：
            The internal structure of the generate() method is as follows: The images for the batch are being loaded, then the given image transformations (if any) are applied to each image and its respective annotations (if any) individually, and once this is done, if a label_encoder was given, then the list of annotations for all images in the batch will be passed to the encoder, which will then return some modified version of those annotations.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple that contains the outputs specified in this set and only those. If an output is not available,
                it will be `None`. The output tuple can contain the following outputs according to the specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                    so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as 'encoded_labels',
                    but containing anchor box coordinates for all matched anchor boxes instead of ground truth coordinates.
                    This can be useful to visualize what anchor boxes are being matched to each ground truth box. Only available
                    in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'evaluation-neutral': A nested list of lists of booleans. Each list contains `True` or `False` for every ground truth
                    bounding box of the respective image depending on whether that bounding box is supposed to be evaluation-neutral (`True`)
                    or not (`False`). May return `None` if there exists no such concept for a given dataset. An example for
                    evaluation-neutrality are the ground truth boxes annotated as "difficult" in the Pascal VOC datasets, which are
                    usually treated to be neutral in a model evaluation.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                    that were applied to the original image to them. This makes it possible to let the model make predictions on a
                    transformed image and then convert these predictions back to the original image. This is mostly relevant for
                    evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                    predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                    original images, not with respect to the transformed images. This means you will have to transform the predicted
                    box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                    image need to be applied in the order in which they are given in the respective list for that image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
                any transformations have been applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
                `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                after they were processed by transformations. Note that the generator checks for degenerate boxes after all
                transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
                are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument.
        '''

        if data_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            # objects_to_shuffle = [data_indice]      # 对于单个元素的列表，sklearn.utils.shuffle的结果不再是列表，而是单个元素本身。只有多于一个的列表才可以返回列表。
            data_indice = sklearn.utils.shuffle(data_indice)

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= data_size:
                current = 0

                #########################################################################################
                # Maybe shuffle the dataset if a full pass over the dataset has finished.
                #########################################################################################

                if shuffle:
                    data_indice = sklearn.utils.shuffle(data_indice)

            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image
            #    files from disk.
            batch_indices = data_indice[current:current + batch_size]
            # 这里的逻辑是：先到内存中寻找数据，然后到hdf5文件中寻找，如果都没有，那么才到硬盘中读取数据.
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[batch_indices]
                else:
                    batch_filenames = None
            elif not (hdf5_dataset is None):
                for i in batch_indices:
                    # 这里在存在hdf5_dataset的情况下，并没有使用hdf5_dataset中的label数据，而是只使用hdf5_dataset中的图像数据，label数据从self.labels中获取.
                    batch_X.append(hdf5_dataset['images'][i].reshape(hdf5_dataset['image_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[batch_indices]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            # Get the labels for this batch (if there are any).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[batch_indices])
            else:
                batch_y = None

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = []  # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Apply any image transformations we may have received. 只对图像做变换，不变换label.
                if transformations:
                    for each in transformations:
                        batch_X[i] = each(batch_X[i])
                '''
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])
                '''

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            batch_X = np.array(batch_X)
            if batch_X.size == 0:
                raise DegenerateBatchError(
                    "You produced an empty batch. This might be because the images in the batch vary " +
                    "in their size and/or number of channels. Note that after all transformations " +
                    "(if any were given) have been applied to all images in the batch, all images " +
                    "must be homogenous in size along all axes.")

            #########################################################################################
            # Compose the output.
            #########################################################################################

            # for debug
            # ret = [batch_X, batch_y, batch_filenames]
            if label_encoder is not None:
                batch_y = label_encoder(batch_y)
            yield batch_X, batch_y

    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None,
                     eval_neutral_path=None):
        '''
        Writes the current `filenames`, `labels`, and `image_ids` lists to the specified files.
        This is particularly useful for large datasets with annotations that are
        parsed from XML files, which can take quite long. If you'll be using the
        same dataset repeatedly, you don't want to have to parse the XML label
        files every time.

        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
            image_ids_path (str, optional): The path under which to save the image IDs pickle.
            eval_neutral_path (str, optional): The path under which to save the pickle for
                the evaluation-neutrality annotations.
        '''
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if not labels_path is None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if not image_ids_path is None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)
        if not eval_neutral_path is None:
            with open(eval_neutral_path, 'wb') as f:
                pickle.dump(self.eval_neutral, f)

    def get_dataset(self):
        '''
        Returns:
            4-tuple containing lists and/or `None` for the filenames, labels, image IDs,
            and evaluation-neutrality annotations.
        '''
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        '''
        Returns:
            The number of images which consists of train and val dataset in the dataset.
        '''
        return self.dataset_size_train, self.dataset_size_val

    @staticmethod
    def test(generator, N):
        '''
        这个函数太棒了！验证通过。
        总结/Trick： 由于数据集模块data_generator最终输出的是生成器，那么不论生成器的中间步骤如何复杂，总可以通过最终这个生成器来验证即将送入网络的训练数据的正确性。
        :param generator:
        :param N: 测试多少张图片.
        :return:
        '''
        for i in range(N):
            rslt = next(generator)
            img = rslt[0][0]
            label = rslt[1][0].flatten()

            print("debugging: ", img.shape, ' ', label)
            cv2.imshow('debug:', img)

            cv2.waitKey(0)


# debug
if __name__ == "__main__":
    '''
    这个脚本的逻辑是： 因为连着看了两天这个数据生成代码，所以整个人现在是懵逼的，但是，必须记录一下，所谓深度在表面！应该即时冒出水面，呼呼新鲜空气。本脚本generator是一个骨架，它将图像数据变换transformation和label编码encoder两个功能结合在了一起。对外提供生成器对象，供模型训练使用。在外部调用的时候，需要首先创建本类对象DataGenerator，然后导入transformation chain，再然后定义label encoder(if any), 最后组装起来，返回生成器对象.
    导入的模块逻辑： 从文件my_augmentation_chain_for_classification_variable_input_size.py中导入入口类，然后这个被导入的文件又借鉴了另外一个chain, 即data_augmentation_chain_base.py，一些几何变换的参数例如随机补黑边，随机裁剪都是在这个文件中设置的。
    '''
    from collections import OrderedDict
    from data_augmentation_chains.my_augmentation_chain_for_classification_variable_input_size import \
        DataAugmentationVariableInputSize as myaug
    from transformations.object_detection_2d_geometric_ops import Resize_arbitrary
    from classification_encoder.input_encoder import InputEncoder

    source_dict = OrderedDict({
        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Notgray', 'index0'): 0,
        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Notgray', 'index1'): 1,
        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Notgray', 'index2'): 2,
        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Isgray', 'index0'): 3,
        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Isgray', 'index1'): 4,
        os.path.join('/home/bro/MyWork/job8_皇室战争/DL/traindata03/inGame/Isgray', 'index2'): 5,
        '/home/bro/MyWork/job8_皇室战争/DL/traindata03/NotinGame': 6,
    })
    train_augmentation_chain = myaug(
        20,
        128,
    )
    val_augmentation_chain = Resize_arbitrary(128,
                                              20,
                                              box_filter=None,
                                              )
    encoder = InputEncoder(7)
    a = DataGenerator()
    a.get_filenames_labels(source_dict, split_ratio=.8)

    # a.create_hdf5_dataset(file_path=['/tmp/train.h5', '/tmp/val.h5', ], )    # 此项目被搁浅，创建hdf5文件不能正确对应label。目前的解决办法是：使用label文件，然后按照这个数据集要求的格式进行组织数据，而不能由着自己的性子来.
    train, val = a.generate(label_encoder=encoder,
                            batch_size=[32, 16], shuffle=[True, False],
                            transformations=[[train_augmentation_chain, ],
                                             [val_augmentation_chain, ]],
                            )

    # train, val = a.generate(source_dict, split_ratio=.8,
    #                         batch_size=[10, 10], shuffle=[True, False],
    #                         transformations=[[train_augmentation_chain, ], [val_augmentation_chain, ]])
    DataGenerator.test(train, 12)
    DataGenerator.test(val, 5)

    # c = next(val)
    # print(c[1], c[2])

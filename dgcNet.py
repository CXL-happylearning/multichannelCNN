import tensorflow as tf
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


class BaseModel(object):
    """Abstract base class for any model."""
    __metaclass__ = ABCMeta

    def __init__(self, num_classes):
        """Constructor.

        Args:
            num_classes: Number of classes.
        """
        self._num_classes = num_classes

    @property
    def num_classes(self):
        return self._num_classes

    @abstractmethod
    def preprocess(self, first_inputs,second_inputs,third_inputs):
        """Input preprocessing. To be override by implementations.

        Args:
            inputs: A float32 tensor with shape [batch_size, height, width,
                num_channels] representing a batch of images.

        Returns:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, widht, num_channels] representing a batch of images.
        """
        pass

    @abstractmethod
    def predict(self, first_preprocessed_inputs,second_preprocessed_inputs,third_preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        pass

    @abstractmethod
    def postprocess(self, prediction_dict, **params):
        """Convert predicted output tensors to final forms.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.

        Returns:
            A dictionary containing the postprocessed results.
        """
        pass

    @abstractmethod
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        pass


class Model(BaseModel):
    """xxx definition."""

    def __init__(self,
                 is_training,
                 num_classes):
        """Constructor.

        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        super(Model, self).__init__(num_classes=num_classes)

        self._is_training = is_training

    def preprocess(self, first_inputs,second_inputs,third_inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        ##tf.to_float改变张量的数据类型，返回一个tensor或sparsetensor
        first_preprocessed_inputs = tf.to_float(first_inputs)
        second_preprocessed_inputs = tf.to_float(second_inputs)
        third_preprocessed_inputs = tf.to_float(third_inputs)

        ##对应元素相减 类型必须是float、int等类型，preprocessed_inputs减去128.0
        first_preprocessed_inputs = tf.subtract(first_preprocessed_inputs, 128.0)
        ##对应元素相除 preprocessed_inputs/128.0
        first_preprocessed_inputs = tf.div(first_preprocessed_inputs, 128.0)

        second_preprocessed_inputs = tf.subtract(second_preprocessed_inputs, 128.0)
        ##对应元素相除 preprocessed_inputs/128.0
        second_preprocessed_inputs = tf.div(second_preprocessed_inputs, 128.0)

        third_preprocessed_inputs = tf.subtract(third_preprocessed_inputs, 128.0)
        ##对应元素相除 preprocessed_inputs/128.0
        third_preprocessed_inputs = tf.div(third_preprocessed_inputs, 128.0)

        return first_preprocessed_inputs,second_preprocessed_inputs,third_preprocessed_inputs

    def predict(self, first_preprocessed_inputs, second_preprocessed_inputs,third_preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
                :param inputs:
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            eyeLeft = first_preprocessed_inputs
            eyeRight = second_preprocessed_inputs
            face = third_preprocessed_inputs

            ###左右眼网络
            ##左眼网络

            eyeLeft_net = slim.conv2d(eyeLeft, 4, [7, 7], scope='eyeLeft_conv1')
            eyeLeft_net = slim.max_pool2d(eyeLeft_net, [2, 2], 2, scope='eyeLeft_pool1')

            eyeLeft_net = slim.conv2d(eyeLeft_net, 8, [5, 5], scope='eyeLeft_conv2')
            eyeLeft_net = slim.max_pool2d(eyeLeft_net, [2, 2], 2, scope='eyeLeft_pool2')

            eyeLeft_net = slim.conv2d(eyeLeft_net, 8, [5, 5], scope='eyeLeft_conv3')
            eyeLeft_net = slim.max_pool2d(eyeLeft_net, [2, 2], 2, scope='eyeLeft_pool3')

            eyeLeft_net = slim.conv2d(eyeLeft_net, 200, [5, 5], scope='eyeLeft_conv4')
            eyeLeft_net = slim.max_pool2d(eyeLeft_net, [2, 2], 2, scope='eyeLeft_pool4')

            ##右眼网络

            eyeRight_net = slim.conv2d(eyeRight, 4, [7, 7], scope='eyeRight_conv1')
            eyeRight_net = slim.max_pool2d(eyeRight_net, [2, 2], 2, scope='eyeRight_pool1')

            eyeRight_net = slim.conv2d(eyeRight_net, 8, [5, 5], scope='eyeRight_conv2')
            eyeRight_net = slim.max_pool2d(eyeRight_net, [2, 2], 2, scope='eyeRight_pool2')

            eyeRight_net = slim.conv2d(eyeRight_net, 8, [5, 5], scope='eyeRight_conv3')
            eyeRight_net = slim.max_pool2d(eyeRight_net, [2, 2], 2, scope='eyeRight_pool3')

            eyeRight_net = slim.conv2d(eyeRight_net, 200, [5, 5], scope='eyeRight_conv4')
            eyeRight_net = slim.max_pool2d(eyeRight_net, [2, 2], 2, scope='eyeRight_pool4')

            #------左右眼网络提取特征图拼接
            eyeLeft = tf.reshape(eyeLeft_net, [-1, int(np.prod(eyeLeft_net.get_shape()[1:]))])
            eyeRight = tf.reshape(eyeRight_net, [-1, int(np.prod(eyeRight_net.get_shape()[1:]))])
            eye_net = tf.concat([eyeLeft, eyeRight], 1)
            #------左右眼网络提取特征拼接

            #全连接特征融合
            eye_net = slim.fully_connected(eye_net, 128, scope='eye_fc1')
            eye_net = slim.dropout(eye_net, 0.5, scope='eye_dropout1')

            ##脸部网络

            face_net = slim.conv2d(face, 4, [7, 7], scope='face_conv1')
            face_net = slim.max_pool2d(face_net, [2, 2], 2, scope='face_pool1')

            face_net = slim.conv2d(face_net, 8, [5, 5], scope='face_conv2')
            face_net = slim.max_pool2d(face_net, [2, 2], 2, scope='face_pool1')

            face_net = slim.conv2d(face_net, 8, [5, 5], scope='face_conv3')
            face_net = slim.max_pool2d(face_net, [2, 2], 2, scope='face_pool1')

            face_net = slim.conv2d(face_net, 8, [5, 5], scope='face_conv4')
            face_net = slim.max_pool2d(face_net, [2, 2], 2, scope='face_pool1')
            #全连接层
            face_net = slim.fully_connected(face_net, 256, scope='face_fc1')
            face_net = slim.dropout(face_net, 0.5, scope='face_dropout1')
            face_net = slim.fully_connected(face_net, 128, scope='face_fc2')
            face_net = slim.dropout(face_net, 0.5, scope='face_dropout2')

            #-----左右眼网络加脸部网络
            face_net = tf.reshape(face_net, [-1, int(np.prod(face_net.get_shape()[1:]))])
            eyeFace_net = tf.concat([eye_net, face_net], 1)
            #-----左右眼网络加脸部网络

            eyeFace_net = slim.fully_connected(eyeFace_net, 128, scope='eyeFace_fc1')
            eyeFace_net = slim.dropout(eyeFace_net, 0.5, scope='eyeFace_dropout1')

            eyeFace_logits = slim.fully_connected(eyeFace_net, self.num_classes,
                                            biases_initializer=tf.zeros_initializer(),
                                            weights_initializer=trunc_normal(1 / 192.0),
                                            weights_regularizer=None,
                                            activation_fn=None,
                                            scope='eyeFace_logits')

            prediction_dict = {'eyeFace_logits': eyeFace_logits}
            return prediction_dict

    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.

        Returns:
            A dictionary containing the postprocessed results.
        """
        #三个通道的网络需要全连接层融合

        eyeFace_logits = prediction_dict['eyeFace_logits']
        eyeFace_logits = tf.nn.softmax(eyeFace_logits)
        logits = eyeFace_logits
        classes = tf.argmax(logits, 1)
        postprecessed_dict = {'classes': classes}
        return postprecessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each image in the batch.

        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        eyeFace_logits = prediction_dict['eyeFace_logits']
        eyeFace_logits = tf.nn.softmax(eyeFace_logits)
        logits = eyeFace_logits
        #softmax只是一个分类器
        slim.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=groundtruth_lists)
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict



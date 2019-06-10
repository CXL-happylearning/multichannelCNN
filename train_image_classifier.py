import tensorflow as tf
from deployment import model_deploy
import dgcNet
import preprocess_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from preprocessing import preprocessing_factory

_PADDING = 4
slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('leftEyeRecord_path',
                    'E:/driveTF5.27/DriverGazeCapture/dataBase/finalData/tfrecord/train_leftEye.tfrecord',
                    'Path to training tfrecord file.')
flags.DEFINE_string('rightEyeRecord_path',
                    'E:/driveTF5.27/DriverGazeCapture/dataBase/finalData/tfrecord/train_rightEye.tfrecord',
                    'Path to training tfrecord file.')
flags.DEFINE_string('faceRecord_path',
                    'E:/driveTF5.27/DriverGazeCapture/dataBase/finalData/tfrecord/train_face.tfrecord',
                    'Path to training tfrecord file.')

flags.DEFINE_string('logdir',
                    'E:/driveTF5.27/DriverGazeCapture/log/train/train/train/',
                    'Path to log directory.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_string(
    'model_name', 'dgcnet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')
FLAGS = flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch FLAGS.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=0.95,
            epsilon=1.0)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=0.1)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=0.9,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
    return optimizer


def main(_):
    # Create global_step
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    with tf.device(deploy_config.inputs_device()):
        datasetLeftEye = preprocess_data.get_record_dataset(FLAGS.leftEyeRecord_path)

        data_provider_leftEye = slim.dataset_data_provider.DatasetDataProvider(
            datasetLeftEye,
            shuffle=True,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        leftEyeImage, leftEyeLabel = data_provider_leftEye.get(['image', 'label'])
        leftEyeLabel -= FLAGS.labels_offset

        lE_training_image_data = tf.image.resize_images(leftEyeImage, [224, 224])
        #train_image_size = FLAGS.train_image_size or network_fn.default_image_size
        #image = image_preprocessing_fn(image, train_image_size, train_image_size)

        leftEyeInputs, leftEyeLabels = tf.train.batch([lE_training_image_data, leftEyeLabel],
                                        batch_size=FLAGS.batch_size,
                                        allow_smaller_final_batch=True,
                                        num_threads=4,
                                        capacity=5 * FLAGS.batch_size)


        datasetRightEye = preprocess_data.get_record_dataset(FLAGS.rightEyeRecord_path)
        data_provider_rightEye = slim.dataset_data_provider.DatasetDataProvider(
            datasetRightEye,
            shuffle=True,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        rightEyeImage, rightEyeLabel = data_provider_rightEye.get(['image', 'label'])
        rightEyeLabel -= FLAGS.labels_offset

        rE_training_image_data = tf.image.resize_images(rightEyeImage, [224, 224])
        # train_image_size = FLAGS.train_image_size or network_fn.default_image_size
        # image = image_preprocessing_fn(image, train_image_size, train_image_size)

        rightEyeInputs, rightEyeLabels = tf.train.batch([rE_training_image_data, rightEyeLabel],
                                                      batch_size=FLAGS.batch_size,
                                                      allow_smaller_final_batch=True,
                                                      num_threads=4,
                                                      capacity=5 * FLAGS.batch_size)



        datasetFace = preprocess_data.get_record_dataset(FLAGS.faceRecord_path)
        data_provider_face = slim.dataset_data_provider.DatasetDataProvider(
            datasetFace,
            shuffle=True,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        faceImage, faceLabel = data_provider_face.get(['image', 'label'])
        faceLabel -= FLAGS.labels_offset

        face_training_image_data = tf.image.resize_images(faceImage, [224, 224])
        # train_image_size = FLAGS.train_image_size or network_fn.default_image_size
        # image = image_preprocessing_fn(image, train_image_size, train_image_size)

        faceInputs, faceLabels = tf.train.batch([face_training_image_data, faceLabel],
                                                        batch_size=FLAGS.batch_size,
                                                        allow_smaller_final_batch=True,
                                                        num_threads=4,
                                                        capacity=5 * FLAGS.batch_size)


        # 将队列中数据打乱后再读取出来
        #image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=10, num_threads=1,
        #                                                  capacity=64, min_after_dequeue=1)


    cls_model = dgcNet.Model(is_training=True, num_classes=7)
    #preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(leftEyeInputs,rightEyeInputs,faceInputs)
    loss_dict = cls_model.loss(prediction_dict, leftEyeLabels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, leftEyeLabels), 'float'))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)

    # 学习率和优化函数设置 根据数据集的数量设置学习率
    learning_rate = _configure_learning_rate(datasetLeftEye.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)

    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir,
                        save_summaries_secs=20, save_interval_secs=120,
                        number_of_steps=110000)


if __name__ == '__main__':
    tf.app.run()

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
import horovod.tensorflow as hvd
import pymysql
import sys
import csv
import shutil
import os
from os.path import expanduser
import time

from tensorflow.python.framework import dtypes
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import importer
from tensorflow.python.tools import freeze_graph


# LSTM Model study : https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
# LSTM Code reference : https://medium.com/@curiousily/human-activity-recognition-using-\
# lstms-on-android-tensorflow-for-hackers-part-vi-492da5adef64

n_time_steps = 20
n_features = 6
n_classes = 6
n_hidden_units = 64
l2_loss = 0.0015
random_seed = 42
learning_rate = 0.0025
batch_size = 1024
# home directory
home = expanduser("~")


def mysql_to_csv(sql, file_path, host, port, user, password, dbName):
    '''
    The function creates a csv file from the result of SQL
    in MySQL database.
    '''
    try:
        con = pymysql.connect(host=host, port=port, user=user, password=password, db=dbName)
        print('Connected to DB: {}'.format(host))
        # Read table with pandas and write to csv
        df = pd.read_sql(sql, con)
        df.to_csv(file_path, encoding='utf-8', header=False, \
                  sep=',', index=False, quoting=csv.QUOTE_NONE)
        print('File, {}, has been created successfully'.format(file_path))
        con.close()

    except Exception as e:
        print('Error: {}'.format(str(e)))
        sys.exit(1)


def create_lstm_model(inputs, target):
    with tf.variable_scope('lstm'):
        w = {
            'hidden': tf.Variable(tf.random_normal([n_features, n_hidden_units])),
            'output': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden_units], mean=1.0)),
            'output': tf.Variable(tf.random_normal([n_classes]))
        }

    x = tf.transpose(inputs, [1, 0, 2])
    x = tf.reshape(x, [-1, n_features])
    hidden = tf.nn.relu(tf.matmul(x, w['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, n_time_steps, 0)

    # Stack 2 LSTM layers
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    pred_y = tf.matmul(lstm_last_output, w['output']) + biases['output']
    pred_softmax = tf.nn.softmax(pred_y, name="y_")

    # loss
    l2 = l2_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=target)) + l2

    return tf.argmax(pred_softmax, 1), loss


def train_input_generator(x_train, y_train, batch_size=64):
    while True:
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def main(_):
    # Horovod: initialize Horovod.
    hvd.init()

    # delete previous saving checkpoints and model
    # if os.path.exists('./checkpoints') and os.path.isdir('./checkpoints'):
    #     shutil.rmtree('./checkpoints')
    if os.path.exists(os.path.join(home, 'data', 'model')) and os.path.isdir(os.path.join(home, 'data', 'model')):
        shutil.rmtree(os.path.join(home, 'data', 'model'))

    # Data set sources : http://archive.ics.uci.edu/ml/datasets/ \
    # Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
    # sensorData_timestamp.txt is pre-processed data and is based on UCI datasets.
    # load dataset from DB
    mysql_to_csv(sql='Select * From sensorData', file_path='./sensorData_timestamp1.csv', host='163.180.117.202',
                 port=3847, user='root', password='password', dbName='hardb')
    columns = ['user', 'activity', 'timestamp', 'acc_x-axis', 'acc_y-axis', 'acc_z-axis', 'gyro_x-axis', 'gyro_y-axis',
               'gyro_z-axis']
    df = pd.read_csv('./sensorData_timestamp1.csv',
                     header=None, names=columns, lineterminator='\n')
    df = df.dropna()

    step = 20
    segments = []
    labels = []
    for i in range(0, len(df) - n_time_steps, step):
        acc_xs = df['acc_x-axis'].values[i: i + n_time_steps]
        acc_ys = df['acc_y-axis'].values[i: i + n_time_steps]
        acc_zs = df['acc_z-axis'].values[i: i + n_time_steps]
        gyro_xs = df['gyro_x-axis'].values[i: i + n_time_steps]
        gyro_ys = df['gyro_y-axis'].values[i: i + n_time_steps]
        gyro_zs = df['gyro_z-axis'].values[i: i + n_time_steps]
        label = stats.mode(df['activity'][i: i + n_time_steps])[0][0]
        segments.append([acc_xs, acc_ys, acc_zs, gyro_xs, gyro_ys, gyro_zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_steps, n_features)
    tmp_df = pd.get_dummies(labels)
    labels = np.asarray(tmp_df, dtype=np.float32)
    reverse_one_hot_encode = tmp_df.idxmax().reset_index().rename(columns={'index': 'activity', 0: 'idx'})
    pickle.dump(reverse_one_hot_encode, open(os.path.join(home, 'data', 'reverse_one_hot_encode'), "wb"))

    # Data split train : test = 80 : 20
    # This split method cause overfit. We need to K-fold taining method.
    x_train, x_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=random_seed)
    pickle.dump(x_test, open(os.path.join(home, 'data', 'x_test'), "wb"))
    pickle.dump(y_test, open(os.path.join(home, 'data', 'y_test'), "wb"))

    # Build model...
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_time_steps, n_features], name="inputs")
        y = tf.placeholder(tf.float32, [None, n_classes], name="label")
    predict, loss = create_lstm_model(x, y)
    tf.summary.scalar("loss", loss)
    # correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Horovod: add Horovod Distributed Optimizer.
    optimizer = hvd.DistributedOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=8000 // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=10),
        tf.train.SummarySaverHook(save_secs=10,
                                  output_dir='/tmp/tf',
                                  summary_op=tf.summary.merge_all())
    ]

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
    training_batch_generator = train_input_generator(x_train, y_train, batch_size=batch_size)
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            input_batch, target = next(training_batch_generator)
            mon_sess.run(train_op, feed_dict={x: input_batch, y: target})

    # save model
    if hvd.rank() != 0:
        return
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    optGraph = optimize_for_inference_lib.optimize_for_inference(tf.get_default_graph().as_graph_def(),
                                                                 ["input/inputs"], ["y_"],
                                                                 dtypes.float32.as_datatype_enum)
    frozenGraph = freeze_graph.freeze_graph_with_def_protos(optGraph, None,
                                                            checkpoint_file, "y_", None, None,
                                                            "frozen.pb", True, None)
    with tf.Graph().as_default():
        importer.import_graph_def(frozenGraph, name="")
        with tf.Session() as sess:
            inputs = tf.get_default_graph().get_tensor_by_name("input/inputs:0")
            model = tf.get_default_graph().get_tensor_by_name("y_:0")
            predictor = tf.argmax(model, 1, name="predictor")
            inputs_classes = tf.saved_model.utils.build_tensor_info(inputs)  # input
            outputs_classes = tf.saved_model.utils.build_tensor_info(predictor)  # output
            signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: inputs_classes},
                outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: outputs_classes},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(home, 'data', 'model'))
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={'predict_activity': signature},
                                                 legacy_init_op=legacy_init_op)
            builder.save()


if __name__ == "__main__":
    tf.app.run()
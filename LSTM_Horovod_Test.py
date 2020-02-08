import tensorflow as tf
import pickle
import os
from os.path import expanduser

# home directory
home = expanduser("~")

# loading
X_TEST = pickle.load(open(os.path.join(home, 'data', 'x_test'), "rb"))
Y_TEST = pickle.load(open(os.path.join(home, 'data', 'y_test'), "rb"))

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], os.path.join(home, 'model'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input/inputs:0")
    model = graph.get_tensor_by_name("predictor:0")
    # print(sess.run(model, {x: [5, 6, 7, 8]}))
    predictor = sess.run(model, {x: X_TEST})

    correct_pred = tf.equal(predictor, tf.argmax(Y_TEST, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    final_correct_pred, final_accuracy = sess.run([correct_pred, accuracy], {x: X_TEST})

    print(final_accuracy)
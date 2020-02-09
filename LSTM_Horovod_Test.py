import tensorflow as tf
import pickle
import os
from os.path import expanduser

# home directory
home = expanduser("~")

# loading
X_TEST = pickle.load(open(os.path.join(home, 'data', 'x_test'), "rb"))
Y_TEST = pickle.load(open(os.path.join(home, 'data', 'y_test'), "rb"))
reverse_one_hot_encode = pickle.load(open(os.path.join(home, 'data', 'reverse_one_hot_encode'), "rb"))
# tf.enable_eager_execution()


def prediction(data_input):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], os.path.join(home, 'data', 'model'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input/inputs:0")
        model = graph.get_tensor_by_name("predictor:0")
        # predictor = sess.run(model, {x: X_TEST[1,:,:][None, :, :]})
        predictor = sess.run(model, {x: X_TEST})
        print(predictor)
        print("reverse_one_hot_encode ", reverse_one_hot_encode['activity'][predictor[0]])
        activity = []
        for i in range(len(predictor)):
            activity.append(reverse_one_hot_encode['activity'][predictor[i]])
        return activity

        # correct_pred = tf.equal(predictor, tf.argmax(Y_TEST, 1))
        # print(correct_pred)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
        #
        # final_correct_pred, final_accuracy = sess.run([correct_pred, accuracy], {x: X_TEST})
        #
        # print(final_accuracy)

activity =  prediction(X_TEST)
print(activity)


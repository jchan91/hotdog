import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def download_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist


def train_simple(mnist_data):
    # Setup model
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    x = tf.placeholder(tf.float32, [None, 784])  # None means dimension of any length (any amount of data)

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Setup training
    y_ = tf.placeholder(tf.float32, [None, 10])

    # loss function
    #
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    #
    # reduction_indices specifies to sum over axis 1 of y_ * log(y)
    #
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(cross_entropy)

    # launch the session
    sess = tf.InteractiveSession()

    # initialize variabls
    init = tf.global_variables_initializer()
    sess.run(init)

    # train
    for i in range(1000):
        print('step {0}'.format(i))
        batch_xs, batch_ys = mnist_data.train.next_batch(100)
        sess.run(train_step, {x: batch_xs, y_: batch_ys})

    # evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, {x: mnist_data.test.images, y_: mnist_data.test.labels}))

    # display
    nW = sess.run(W)
    W0 = np.reshape(nW[:, 0], (28, 28))
    plt.imshow(W0)
    plt.show()


def train_estimator():
    feature_columns = [tf.feature_columns.numeric_column('x', shape=[784])]

    estimator = tf.estimator.LinearRegressor(
        feature_columns=feature_columns)


if __name__ == "__main__":
    mnist_data = download_mnist()
    train_simple(mnist_data)

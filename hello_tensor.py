import tensorflow as tf
import numpy as np


def sample1():
    node1 = tf.constant(3.0)
    node2 = tf.constant(4.0)

    sess = tf.Session()

    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print(node3)
    print(sess.run(node3))


def sample2(sess):
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b

    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

    add_and_triple = adder_node * 3

    print(sess.run(add_and_triple, {a: [1, 3], b: [2, 4]}))


def sample_linear_model(
    sess
):
    # Variable is for trainable parameters
    # placeholder is for inputs/outputs (e.g. image pixel values)
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)

    linear_model = W * x + b

    # need this to initialize all Variables
    # placeholders are provided on run
    init = tf.global_variables_initializer()
    sess.run(init)

    inputs = {x: [1, 2, 3, 4]}
    print(sess.run(linear_model, inputs))


def sample_loss(
    sess
):
    # model f(x) = Wx + b
    # loss = squared_sum(y - f(x))
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # initialize linear_model
    init = tf.global_variables_initializer()
    sess.run(init)

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


def sample_update(
    sess
):
    # model f(x) = Wx + b
    # loss = squared_sum(y - f(x))
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # initialize linear_model
    init = tf.global_variables_initializer()
    sess.run(init)

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


def sample_optimize(
    sess
):
    # model f(x) = Wx + b
    # loss = squared_sum(y - f(x))
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    # create an optimizer set to optimize 'loss'
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # initialize linear_model
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([W, b]))


def sample_full():
    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


def sample_estimator():
    # Declare list of features. We only have one numeric feature. There are many
    # other types of columns that are more complicated and useful.
    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

    # An estimator is the front end to invoke training (fitting) and evaluation
    # (inference). There are many predefined types like linear regression,
    # linear classification, and many neural network classifiers and regressors.
    # The following code provides an estimator that does linear regression.
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use two data sets: one for training and one for evaluation
    # We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

    # We can invoke 1000 training steps by invoking the  method and passing the
    # training data set.
    estimator.train(input_fn=input_fn, steps=1000)

    # Here we evaluate how well our model did.
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % eval_metrics)


if __name__ == "__main__":
    sess = tf.Session()
    sample_estimator()

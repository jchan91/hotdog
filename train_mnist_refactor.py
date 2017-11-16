import tensorflow as tf
import time
import numpy as np


def download_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def setup_placeholders(
    num_pixels,
    num_labels,
    batch_size
):
    '''
    Returns:
    img_placeholder: Has shape=(None, img_numPixels)
    label_placeholder: Has shape=(None, num_labels)

    '''
    img_placeholder = tf.placeholder(tf.float32, shape=[batch_size, num_pixels])
    label_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    dropout_placeholder = tf.placeholder(tf.float32)
    return img_placeholder, label_placeholder, dropout_placeholder


def fill_feed_dict(
    dataset,
    img_placeholder,
    label_placeholder,
    dropout_placeholder,
    batch_size=50,
    dropout=0.5
):
    '''
    Args:
        dataset: Data provider that has images and labels. Must implement method, next_batch()
        img_placeholder
        label_placeholder
    '''
    images, labels = dataset.next_batch(batch_size)
    labels = np.argmax(labels, 1)
    return {
        img_placeholder: images,
        label_placeholder: labels,
        dropout_placeholder: dropout
    }


def setup_inference(
    img_placeholder,
    dropout_placeholder
):
    x_image = tf.reshape(img_placeholder, [-1, 28, 28, 1])

    # Conv 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Conv 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = weight_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # FC1
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 50 x 3136
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout_placeholder)

    # FC2
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y


def setup_loss(
    logits,
    labels
):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='xentropy')
    return tf.reduce_mean(
        cross_entropy,
        name='xentropy_mean')


def setup_training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def setup_evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def run_training():
    data_sets = download_mnist()
    batch_size = 50
    dropout_prob = 0.5

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder, dropout_placeholder = setup_placeholders(
            num_pixels=784,
            num_labels=10,
            batch_size=batch_size)

        logits = setup_inference(
            images_placeholder,
            dropout_placeholder)

        loss = setup_loss(logits, labels_placeholder)

        train_op = setup_training(loss, 0.1)

        eval_correct = setup_evaluation(
            logits,
            labels_placeholder)

        summary = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        log_dir = 'C:/data/temp/'
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        for step in range(20000):
            start_time = time.time()

            feed_dict = fill_feed_dict(
                data_sets.train,
                images_placeholder,
                labels_placeholder,
                dropout_placeholder,
                batch_size,
                dropout_prob)

            _, loss_value = sess.run(
                [train_op, loss],
                feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()


if __name__ == "__main__":
    run_training()

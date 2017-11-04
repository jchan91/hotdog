import tensorflow as tf


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


def linear_model(
    sess
):
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)

    linear_model = W * x + b

    inputs = {x: [1, 2, 3, 4]}
    print(sess.run(linear_model, inputs))


if __name__ == "__main__":
    sess = tf.Session()
    linear_model(sess)

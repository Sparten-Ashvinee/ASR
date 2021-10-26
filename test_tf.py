#
#
# def test_convnet():
#   image = tf.placeholder(tf.float32, (None, 100, 100, 3)
#   model = Model(image)
#   sess = tf.Session()
#   sess.run(tf.global_variables_initializer())
#   before = sess.run(tf.trainable_variables())
#   _ = sess.run(model.train, feed_dict={
#                image: np.ones((1, 100, 100, 3)),
#                })
#   after = sess.run(tf.trainable_variables())
#   for b, a in zip(before, after):
#       # Make sure something changed.
#       assert (b != a).any()
#
#   def make_convnet(image_input):
#       # Try to normalize the input before convoluting
#       net = slim.batch_norm(image_input)
#       net = slim.conv2d(net, 32, [11, 11], scope="conv1_11x11")
#       net = slim.conv2d(net, 64, [5, 5], scope="conv2_5x5")
#       net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool1')
#       net = slim.conv2d(net, 64, [5, 5], scope="conv3_5x5")
#       net = slim.conv2d(net, 128, [3, 3], scope="conv4_3x3")
#       net = slim.max_pool2d(net, [2, 2], scope='pool2')
#       net = slim.conv2d(net, 128, [3, 3], scope="conv5_3x3")
#       net = slim.max_pool2d(net, [2, 2], scope='pool3')
#       net = slim.conv2d(net, 32, [1, 1], scope="conv6_1x1")
#       return net
#
#
# class Model:
#   def __init__(self, input, labels):
#     """Classifier model
#     Args:
#       input: Input tensor of size (None, input_dims)
#       label: Label tensor of size (None, 1).
#         Should be of type tf.int32.
#     """
#     prediction = self.make_network(input)
#     # Prediction size is (None, 1).
#     self.loss = tf.nn.softmax_cross_entropy_with_logits(
#         logits=prediction, labels=labels)
#     self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
#
#
# def test_loss():
#   in_tensor = tf.placeholder(tf.float32, (None, 3))
#   labels = tf.placeholder(tf.int32, None, 1))
#   model = Model(in_tensor, labels)
#   sess = tf.Session()
#   loss = sess.run(model.loss, feed_dict={
#     in_tensor:np.ones(1, 3),
#     labels:[[1]]
#   })
#   assert loss != 0
#
#
# class GAN:
#   def __init__(self, z_vector, true_images):
#     # Pretend these are implemented.
#     with tf.variable_scope("gen"):
#       self.make_geneator(z_vector)
#     with tf.variable_scope("des"):
#       self.make_descriminator(true_images)
#     opt = tf.AdamOptimizer()
#     train_descrim = opt.minimize(self.descrim_loss)
#     train_gen = opt.minimize(self.gen_loss)
#
#
#
# def test_gen_training():
#   model = Model
#   sess = tf.Session()
#   gen_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
#   des_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='des')
#   before_gen = sess.run(gen_vars)
#   before_des = sess.run(des_vars)
#   # Train the generator.
#   sess.run(model.train_gen)
#   after_gen = sess.run(gen_vars)
#   after_des = sess.run(des_vars)
#   # Make sure the generator variables changed.
#   for b,a in zip(before_gen, after_gen):
#     assert (a != b).any()
#   # Make sure descriminator did NOT change.
#   for b,a in zip(before_des, after_des):
#     assert (a == b).all()
#
#
#

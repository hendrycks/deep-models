import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import time
import pickle
import os


def opts_parser():
  usage = "Trains and tests a DenseNet on CIFAR-10."
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '-L', '--depth', type=int, default=40,
    help='Network depth in layers (default: %(default)s)')
  parser.add_argument(
    '-k', '--growth-rate', type=int, default=12,
    help='Growth rate in dense blocks (default: %(default)s)')
  parser.add_argument(
    '--dropout', type=float, default=0,
    help='Dropout rate (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Perform data augmentation (not enabled)')
  parser.add_argument(
    '--no-augment', action='store_false', dest='augment',
    help='Disable data augmentation')
  parser.add_argument(
    '--epochs', type=int, default=300,
    help='Number of training epochs (default: %(default)s)')
  parser.add_argument(
    '--save-weights', type=str, default=None, metavar='FILE',
    help='If given, save network weights to given .npz file')
  parser.add_argument(
    '--save-errors', type=str, default=None, metavar='FILE',
    help='If given, save train/validation errors to given .npz file')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Should we continue training from some earlier model?')
  parser.add_argument(
    '--nonlinearity_name', type=str, default="relu",
    help='Nonlinearity name (relu default)')
  parser.add_argument(
    '--use_cifar10', action='store_true', default=False,
    help='Use CIFAR-10')
  parser.add_argument(
    '--use_cifar100', action='store_true', default=False,
    help='Use CIFAR-100')
  parser.add_argument(
    '--use_svhn', action='store_true', default=False,
    help='Use SVHN')
  return parser


def batch_norm(x, phase_train, scope='bn'):
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=x.get_shape()[-1:]),
                       name='beta', trainable=True)
    with tf.variable_scope("regularize"):
      gamma = tf.Variable(tf.constant(1.0, shape=x.get_shape()[-1:]),
                          name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-4)
  return normed


def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):
  res = [0] * len(tensors)
  batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in batch_placeholders]
  total_size = len(batch_tensors[0][1])
  batch_count = (total_size + batch_size - 1) // batch_size
  for batch_idx in range(batch_count):
    current_batch_size = None
    for (placeholder, tensor) in batch_tensors:
      batch_tensor = tensor[batch_idx*batch_size : (batch_idx+1)*batch_size]
      current_batch_size = len(batch_tensor)
      feed_dict[placeholder] = tensor[batch_idx*batch_size : (batch_idx+1)*batch_size]
    tmp = session.run(tensors, feed_dict=feed_dict)
    res = [r + t * current_batch_size for (r, t) in zip(res, tmp)]
  return [r / total_size for r in res]


def weights_and_biases(kernel_shape, bias_shape,name=""):
  # Create variable named "weights".
  weights = tf.get_variable("weights" + name, kernel_shape,
          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1/np.sqrt(kernel_shape[0]*0.5)))
          # initializer=tf.contrib.layers.xavier_initializer())
  tf.add_to_collection("regularize", weights)
  # Create variable named "biases".
  biases = tf.get_variable("biases" + name, bias_shape,
          # initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
          initializer=tf.constant_initializer(value=0, dtype=tf.float32))
  return weights, biases


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W, B =  weights_and_biases([kernel_size, kernel_size, in_features, out_features], [out_features])
  conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
  # if with_bias:
  #   return conv + B
  return conv


def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob, f=tf.nn.relu):
  current = batch_norm(current, is_training)
  current = f(current)
  current = conv2d(current, in_features, out_features, kernel_size)
  current = tf.nn.dropout(current, keep_prob)
  return current


def block(input, layers, in_features, growth, is_training, keep_prob, f=tf.nn.relu):
  current = input
  features = in_features
  for idx in range(layers):
    with tf.variable_scope("conv"+str(idx)):
      tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob, f)
      current = tf.concat(3, (current, tmp))
      features += growth
  return current, features


def avg_pool(input, s):
  return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')


def run_model(data, label_count, depth=40, growth_rate=12, dropout=0, epochs=300,
              save_weights='', save_errors='', resume=False, f=tf.nn.relu):
  weight_decay = 1e-4
  layers = (depth - 4) // 3
  graph = tf.Graph()
  with graph.as_default():
    xs = tf.placeholder("float", shape=[None, 32, 32, 3])
    ys = tf.placeholder("int64", shape=[None])
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])

    # current = tf.reshape(xs, [-1, 32, 32, 3])
    # current = conv2d(current, 3, 16, 3)
    current = conv2d(xs, 3, 16, 3)

    with tf.variable_scope("block1"):
      current, features = block(current, layers, 16, growth_rate, is_training, keep_prob, f)
      current = batch_activ_conv(current, features, features, 1, is_training, keep_prob, f)
      current = avg_pool(current, 2)
    with tf.variable_scope("block2"):
      current, features = block(current, layers, features, growth_rate, is_training, keep_prob, f)
      current = batch_activ_conv(current, features, features, 1, is_training, keep_prob, f)
      current = avg_pool(current, 2)  # why not just l2 pool?
    with tf.variable_scope("block3"):
      current, features = block(current, layers, features, growth_rate, is_training, keep_prob, f)

      current = batch_norm(current, is_training)
    with tf.variable_scope("fc"):
      current = f(current)
      current = avg_pool(current, 8)
      final_dim = features
      current = tf.reshape(current, [-1, final_dim])
      #Wfc = weight_variable([final_dim, label_count])
      #bfc = bias_variable([label_count])
      Wfc, bfc =  weights_and_biases([final_dim, label_count], [label_count])
      logits = tf.matmul(current, Wfc) + bfc


  with tf.Session(graph=graph) as session:
    batch_size = 64
    learning_rate = 0.1

    # we place these loss components here so we can select the variables to regularize easily
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.get_collection("regularize")])
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(logits, 1), ys)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    start_epoch = 1
    errors = []
    if resume:
      errors = list(np.load(save_errors+'.npy'))
      for i in range(epochs-1,-1,-1):
        try:
          saver.restore(session, save_weights + '_%d.ckpt' % i)
          start_epoch = i+1
          print('Restored!')
          break
        except:
          True
      if start_epoch == 1:
        assert False, "could not resume"

    train_data, train_labels = data['train_data'], data['train_labels']
    batch_count = len(train_data) // batch_size
    # batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
    # batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
    print("Batch per epoch: ", batch_count)
    for epoch in range(start_epoch, 1+epochs):
      t_start = time.time()
      indices = np.arange(train_data.shape[0])
      np.random.shuffle(indices)
      train_data = train_data[indices]
      train_labels = train_labels[indices]
      if epoch == epochs//2: learning_rate = 0.01
      if epoch == 3*epochs//4: learning_rate = 0.001
      for batch_idx in range(batch_count):
        offset = batch_size * batch_idx
        xs_, ys_ = train_data[offset:offset + batch_size], train_labels[offset:offset + batch_size]
        batch_res = session.run([train_step, cross_entropy, accuracy],
                                feed_dict={xs: xs_, ys: ys_, lr: learning_rate, is_training: True, keep_prob: 1 - dropout})
        if batch_idx % (batch_count//2) == 0:
          errors.append(batch_res[1:])
          print(epoch, batch_idx, batch_res[1:])
      print('Took', time.time() - t_start)
      saver.save(session, save_weights + '_%d.ckpt' % epoch)
      test_results = run_in_batch_avg(session, [cross_entropy, accuracy], [xs, ys],
                                      feed_dict={xs: data['test_data'], ys: data['test_labels'],
                                                 is_training: False, keep_prob: 1.})
      errors.append(test_results)
      np.save(save_errors, errors)
      print('Test results', epoch, batch_res[1:], test_results)


def train_test(depth, growth_rate, dropout, augment, epochs,
               save_weights, save_errors, resume, nonlinearity_name,
               use_cifar10, use_cifar100, use_svhn):

  if use_cifar10:
    label_count = 10

    data = [pickle.load(open(os.path.join('/home-nfs/dan/cifar_data/cifar-10-batches-py',
                                          'data_batch_%d' % (i + 1)), 'rb'), encoding='latin1') for i in range(5)]
    train_data = np.vstack([d['data'] for d in data])
    train_labels = np.hstack([np.asarray(d['labels'], np.int8) for d in data])

    data = pickle.load(open(os.path.join('/home-nfs/dan/cifar_data/cifar-10-batches-py', 'test_batch'),
                            'rb'), encoding='latin1')
    test_data = data['data']
    test_labels = np.asarray(data['labels'], np.int8)

    train_data = np.dstack((train_data[:, :1024], train_data[:, 1024:2048], train_data[:, 2048:]))
    test_data = np.dstack((test_data[:, :1024], test_data[:, 1024:2048], test_data[:, 2048:]))

  elif use_cifar100:
    label_count = 100

    data = pickle.load(open(os.path.join('/home-nfs/dan/cifar_data/cifar-100-python', "train"),
                            'rb'), encoding='latin1')
    train_data = data['data']
    train_labels = np.asarray(data['fine_labels'], np.int8)

    data = pickle.load(open(os.path.join('/home-nfs/dan/cifar_data/cifar-100-python', 'test'),
                            'rb'), encoding='latin1')
    test_data = data['data']
    test_labels = np.asarray(data['fine_labels'], np.int8)

    train_data = np.dstack((train_data[:, :1024], train_data[:, 1024:2048], train_data[:, 2048:]))
    test_data = np.dstack((test_data[:, :1024], test_data[:, 1024:2048], test_data[:, 2048:]))

  elif use_svhn:
    label_count = 10
    import scipy.io as sio

    # training data
    data = sio.loadmat(os.path.join('/home-nfs/dan/cifar_data/svhn', "train_32x32.mat"))
    X_train = np.rollaxis(data['X'], 3)
    y_train = np.asarray(np.squeeze(data['y']) - 1, np.int8)    # they start counting from 1, so we subtract

    data = sio.loadmat(os.path.join('/home-nfs/dan/cifar_data/svhn', "extra_32x32.mat"))
    train_data = np.concatenate((X_train, np.rollaxis(data['X'], 3)))
    # they start counting from 1, so we subtract
    train_labels = np.concatenate((y_train, np.asarray(np.squeeze(data['y']) - 1, np.int8)))

    del X_train, y_train

    # test data
    data = sio.loadmat(os.path.join('/home-nfs/dan/cifar_data/svhn', 'test_32x32.mat'))
    test_data = np.rollaxis(data['X'], 3)
    test_labels = np.asarray(np.squeeze(data['y']) - 1, np.int8)

  else:
    assert False, "need dataset to be specified"

  train_data = train_data.reshape(-1, 32, 32, 3)
  test_data = test_data.reshape(-1, 32, 32, 3)

  mean = train_data.mean(axis=(0,), keepdims=True).astype(np.float32)
  std = train_data.std(axis=(0,), keepdims=True).astype(np.float32)

  train_data = (train_data - mean) / std
  test_data = (test_data - mean) / std

  if nonlinearity_name == 'relu':
    f = tf.nn.relu
  elif nonlinearity_name == 'elu':
    f = tf.nn.elu
  elif nonlinearity_name == 'gelu':
    def gelu(x):
      return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    f = gelu
  else:
    assert False, 'Need valid nonlinearity inputted'


  print("Train:", np.shape(train_data), np.shape(train_labels))
  print("Test:", np.shape(test_data), np.shape(test_labels))
  data = {'train_data': train_data,
          'train_labels': train_labels,
          'test_data': test_data,
          'test_labels': test_labels}
  run_model(data, label_count, depth, growth_rate, dropout, epochs,
            save_weights, save_errors, resume, f)


def main():
  # parse command line
  parser = opts_parser()
  args = parser.parse_args()

  # run
  train_test(**vars(args))


if __name__ == "__main__":
  main()

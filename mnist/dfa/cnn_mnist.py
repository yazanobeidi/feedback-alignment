from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dfa as DFA

model_save_dir = "/tmp/mnist_dfa"

tf.logging.set_verbosity(tf.logging.INFO)

# Number of epochs (iteration over 1000 images) to train for
TRAIN_EPOCHS = 50

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = input_layer = tf.reshape(features["x"], [-1, 784])
  
  # First dense layer
  # 200 nodes
  with tf.variable_scope('dense1') as scope:
    dim = input_layer.get_shape()[1].value
    weights = tf.get_variable(name='weights',
                              shape=[dim, 200],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32))
    biases = tf.get_variable(name='biases',
                             shape=[200],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    pre_activation = tf.add(tf.matmul(input_layer, weights), biases, name='pre_activation')
    dense1 = tf.nn.tanh(pre_activation, name='activations')

  # Second dense layer
  # 100 nodes
  with tf.variable_scope('dense2') as scope:
    dim = dense1.get_shape()[1].value
    weights = tf.get_variable(name='weights',
                              shape=[dim, 100],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32))
    biases = tf.get_variable(name='biases',
                             shape=[100],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    pre_activation = tf.add(tf.matmul(dense1, weights), biases, name='pre_activation')
    dense2 = tf.nn.tanh(pre_activation, name='activations')


  # Logits layer
  with tf.variable_scope('logits') as scope:
    dim = dense2.get_shape()[1].value
    weights = tf.get_variable(name='weights',
                              shape=[dim, 10],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32))
    biases = tf.get_variable(name='biases',
                             shape=[10],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    pre_activation = tf.add(tf.matmul(dense2, weights), biases, name='pre_activation')
    logits = tf.add(pre_activation, 0.0, name='activations')

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = DFA.DFA(learning_rate=0.5)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_save_dir)
  train_log = []
  test_log = []
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  for i in range(0, TRAIN_EPOCHS):
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=1000,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    # Evaluate the model on the first 10000 training examples and print results
    eval_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data[:10000]},
        y=train_labels[:10000],
        num_epochs=1,
        shuffle=False)
    train_eval_results = mnist_classifier.evaluate(input_fn=eval_train_input_fn)
    train_log.append(train_eval_results)
    print("Training Data Performance: (first 10000 examples)\n", train_eval_results)

    # Evaluate the model on the evaluation data and print results
    eval_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    test_eval_results = mnist_classifier.evaluate(input_fn=eval_test_input_fn)
    test_log.append(test_eval_results)
    print("Testing Data Performance:\n", test_eval_results)
  train_loss = [x['loss'] for x in train_log]
  train_acc = [x['accuracy'] for x in train_log]
  test_loss = [x['loss'] for x in test_log]
  test_acc = [x['accuracy'] for x in test_log]
  line_train_loss = plt.plot(train_loss, label="Training Set")
  line_test_loss = plt.plot(test_loss, label="Testing Set")
  plt.legend()
  plt.show()
  line_train_acc = plt.plot(train_acc, label="Training Set")
  line_test_acc = plt.plot(test_acc, label="Testing Set")
  plt.legend()
  plt.show()

if __name__ == "__main__":
  tf.app.run()

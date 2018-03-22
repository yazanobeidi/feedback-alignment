from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer, training_ops
import tensorflow as tf
import numpy as np

class DFA(optimizer.Optimizer): # A lot copy-pasted from the optimizer base class, with anything that gave errors taken out...

    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2
    
    def __init__(self, learning_rate=0.001, stddev=0.5, use_locking=False, name="DFA"):
        super(DFA, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._stddev = stddev # not really because it's a uniform distribution, [-0.5, 0.5] was the value lillicrap used
        self._B = [] # Build this the first time compute_gradients is called so the shape of each layer can be used

        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate") # Normally these are in _prepare but it seems that doesn't get called before compute_gradients for this implementation?
        self._stddev_t = ops.convert_to_tensor(self._stddev, name="standard_deviation")

    def _prepare(self): # idk
        pass

    def compute_gradients(self, loss, var_list=None, # Copy-paste and black magic
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        if callable(loss):
          with backprop.GradientTape() as tape:
            if var_list is not None:
              tape.watch(var_list)
            loss_value = loss()
          if var_list is None:
            var_list = tape.watched_variables()
          grads = tape.gradient(loss_value, var_list, grad_loss)
          return list(zip(grads, var_list))
        if gate_gradients not in [optimizer.Optimizer.GATE_NONE, optimizer.Optimizer.GATE_OP,
                                  optimizer.Optimizer.GATE_GRAPH]:
          raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                           "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                           gate_gradients)
        self._assert_valid_dtypes([loss])
        if grad_loss is not None:
          self._assert_valid_dtypes([grad_loss])
        if var_list is None:
          var_list = (
              tf.trainable_variables() +
              ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        else:
          var_list = nest.flatten(var_list)
        # pylint: disable=protected-access
        var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
        # pylint: enable=protected-access
        if not var_list:
          raise ValueError("No variables to optimize.")
        B = [] # so it's in the right scope
        if(self._B == []): # if it hasn't been initialized
            stddev = math_ops.cast(self._stddev_t, tf.float32)
            for v in var_list: # iterate through layers and give B the same shape
                B.append(tf.random_uniform(v.get_shape(), minval=-1*stddev, maxval=stddev, dtype=tf.float32)) # random matrix, uniform distribution between [-stddev, stddev]
            self._B = B
        else:
            B = self._B
        grads = [tf.multiply(loss, B[i]) for i in range(0,len(var_list))] # matmul didn't work but I guess this does
        
        if gate_gradients == optimizer.Optimizer.GATE_GRAPH:
          grads = control_flow_ops.tuple(grads)
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars
             if g is not None and v.dtype != dtypes.resource])
        return grads_and_vars

    def _apply_dense(self, grad, var): # literally(ish) copy-pasted from tensorflow's stochastic gradient descent implementation, which is mostly hooks to C++, should deal with derivatives for us. Can swap this for something like RMSProp, Adam, etc pretty easily too.
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op

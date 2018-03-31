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
        self._stddev = stddev
        self._B = []

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
        processors = [optimizer._get_processor(v) for v in var_list]
        if not var_list:
          raise ValueError("No variables to optimize.")
        var_refs = [p.target() for p in processors]
        act_refs = []
        if(self._B == []):
            stddev = math_ops.cast(self._stddev_t, tf.float32)
            for v in var_refs:
                shape = [10]+v.get_shape().as_list()
                shape.reverse()
                if(self._stddev == 0):
                    stddev = tf.rsqrt(math_ops.cast(v.shape[-1], tf.float32))
                self._B.append(tf.Variable(tf.random_uniform(shape, minval=tf.multiply(-1., stddev), maxval=stddev), trainable=False, name=v.op.name+'/B')) # random matrix, uniform distribution between [-stddev, stddev]
                act_name = "/".join(v.op.name.split("/")[:-1])+"/activations"
                act_refs.append(tf.get_default_graph().get_operation_by_name(act_name).outputs[0])
        f_grad = tf.reduce_mean(tf.gradients(loss, act_refs[-1])[0], axis=0)
        grads = [tf.multiply(tf.reduce_sum(tf.transpose(tf.multiply(f_grad, self._B[i])),axis=0),
                             tf.gradients(act_refs[i], var_refs[i], grad_ys=grad_loss, gate_gradients=(gate_gradients == optimizer.Optimizer.GATE_OP), aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops)[0])
                 for i in range(0,len(var_refs)-2)]
        grads.append(tf.gradients(loss, var_refs[-2],
                                  grad_ys=grad_loss,
                                  gate_gradients=(gate_gradients == optimizer.Optimizer.GATE_OP),
                                  aggregation_method=aggregation_method,
                                  colocate_gradients_with_ops=colocate_gradients_with_ops)[0])
        grads.append(tf.gradients(loss, var_refs[-1],
                                  gate_gradients=(gate_gradients == optimizer.Optimizer.GATE_OP),
                                  aggregation_method=aggregation_method,
                                  colocate_gradients_with_ops=colocate_gradients_with_ops)[0]) # Get the gradients for the final layer as tensorflow does normally(?)
        if gate_gradients == optimizer.Optimizer.GATE_GRAPH:
          grads = control_flow_ops.tuple(grads)
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars
             if g is not None and v.dtype != dtypes.resource])
        return grads_and_vars

    def _apply_dense(self, grad, var):
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op

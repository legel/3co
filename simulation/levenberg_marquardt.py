from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from tensorflow_graphics.util import export_api


def _internal2external_grad(parameter_values, lower_bound=-0.000001, upper_bound=1.000001):
  """
  Calculate the internal (unconstrained) to external (constained) parameter gradiants.
  """
  i2e_gradients = (upper_bound - lower_bound) * tf.math.cos(parameter_values) / 2.0
  return i2e_gradients

def _external2internal_grad(parameter_values, lower_bound=-0.000001, upper_bound=1.000001):
  # derivative of arcsin(x) = sqrt(1/(1-x**2))
  x = (2.0 * (parameter_values - lower_bound) / (upper_bound - lower_bound)) - 1.

  print("DENOM OF GRAD SCALE:")
  print((1-x**2))

  e2i_gradients = tf.math.sqrt(1/(1-x**2))
  return e2i_gradients

def _internal2external_params(parameter_values, lower_bound=-0.000001, upper_bound=1.000001):
  i2e_parameters = lower_bound + ((upper_bound - lower_bound) / 2.) * (tf.math.sin(parameter_values) + 1.0)
  return i2e_parameters


def _external2internal_params(parameter_values, lower_bound=-0.000001, upper_bound=1.000001):
  return tf.math.asin((2.0 * (parameter_values - lower_bound) / (upper_bound - lower_bound)) - 1.)


#### We've applied a bounding function, and try to have it so that the optimization proceeds unbounded,
#### while the external valus get clipped along a sinusoidal clipper, with gradients included (somewhere?)
#### There are a few challenges:
####  - Disentangling the photometric_loss(hypothesis_brdf) from calling renders, etc. with gradients; just need a pure loss f()
####  - Are the gradients applied?  
####  - Are the scaled values reasonable - e.g. why does the scaled values of 0.0 convert to 0.5?


def _loss_parameters_and_jacobian(loss_functions, internal_variables):
  """
  Computes the loss values, latest scaled parameters, and the scaled Jacobian matrix.
  """
  # get loss function
  loss_function = loss_functions[0]

  # get parameters scaled for computing external values (e.g. inverse rendering loss)
  external_variables = _internal2external_params(internal_variables)

  print("EXTERNAL VARIABLES GOING INTO RENDER LOSS (PRIOR TO RESHAPE):")
  print(external_variables)
  print(external_variables.shape)


  external_variables = tf.squeeze(external_variables)
  print("EXTERNAL VARIABLES GOING INTO RENDER LOSS (AFTER SQUEEZE):")
  print(external_variables)
  print(external_variables.shape)

  # compute loss and get gradients
  loss_outputs, external_gradients = loss_function(external_variables)

  # transform into proper data structures
  loss_outputs = tf.expand_dims([loss_outputs], axis=1)
  external_gradients = tf.expand_dims(external_gradients, axis=-1)
  external_gradients = tf.transpose(external_gradients)

  print("EXTERNAL GRADIENTS:")
  print(external_gradients)

  internal_gradients_scale = _external2internal_grad(external_variables)

  print("GRADIENT SCALE:")
  print(internal_gradients_scale)

  # scale gradients 
  # external_gradient_scale = _internal2external_grad(internal_variables)
  internal_gradients = internal_gradients_scale * external_gradients
  print("INTERNAL GRADIENTS")
  print(internal_gradients)

  return loss_outputs, internal_gradients


def minimize(loss_functions,
             variables,
             max_iterations=100,
             regularizer=1e-10,
             regularizer_multiplier=10.0,
             callback=None,
             name="levenberg_marquardt_minimize"):
  r"""Minimizes a set of loss_functions in the least-squares sense.
  """
  if not isinstance(variables, (tuple, list)):
    variables = [variables]
  with tf.name_scope(name):
    if not isinstance(loss_functions, (tuple, list)):
      loss_functions = [loss_functions]
    if isinstance(loss_functions, tuple):
      loss_functions = list(loss_functions)
    if isinstance(variables, tuple):
      variables = list(variables)
    variables = [tf.convert_to_tensor(value=variable) for variable in variables]
    multiplier = tf.constant(regularizer_multiplier, dtype=variables[0].dtype)

    if max_iterations <= 0:
      raise ValueError("'max_iterations' needs to be at least 1.")

    def _cond(iteration, regularizer, objective_value, variables):
      """Returns whether any iteration still needs to be performed."""
      del regularizer, objective_value, variables
      return iteration < max_iterations

    def _levenberg_marquardt_optimization_step(iteration, regularizer, original_objective_value, internal_variables):
      # update iteration
      iteration += tf.constant(1, dtype=tf.int32)

      # compute objective function values and internal gradients for current hypothesis
      objective_value, internal_gradients = _loss_parameters_and_jacobian(loss_functions, internal_variables)


      print(internal_gradients)
      print(objective_value)

      # compute updates in internal Levenberg Marquardt unbounded space
      updates = tf.linalg.lstsq(internal_gradients, objective_value, fast=False) # l2_regularizer=regularizer, 

      # work with original variables where possible
      shapes = [tf.shape(input=variable) for variable in internal_variables]
      splits = [tf.reduce_prod(input_tensor=shape) for shape in shapes]
      updates = tf.split(tf.squeeze(updates, axis=-1), splits)

      # apply update by subtracting update value from variable
      new_internal_variables = [variable + tf.reshape(update, shape) for variable, update, shape in zip(internal_variables, updates, shapes)]
      new_internal_variables = tf.concat(new_internal_variables, axis=0)

      # check how well the update did
      new_objective_value, _, = _loss_parameters_and_jacobian(loss_functions, new_internal_variables)

      # If the new estimated solution does not decrease the objective value,
      # no updates are performed, but a new regularizer is computed.
      cond = tf.less(new_objective_value, objective_value)
      # regularizer = tf.where(cond, x=regularizer, y=regularizer * multiplier)

      # objective_value_to_return = tf.where(cond, x=new_objective_value, y=objective_value)
      # internal_variables_to_return = [tf.where(cond, x=new_variable, y=variable) for variable, new_variable in zip(internal_variables, new_internal_variables)]
      internal_variables_to_return = tf.concat(new_internal_variables, axis=0)

      return iteration, regularizer, new_objective_value, new_internal_variables

    # initialize optimization
    loss_function = loss_functions[0]
    objective_value, _ = loss_function(*variables)
    dtype = variables[0].dtype

    internal_variables = _external2internal_params(variables[0])

    initial = (tf.constant(0, dtype=tf.int32), tf.constant(regularizer, dtype=dtype), objective_value, internal_variables)

    _, _, final_objective_value, final_variables = tf.while_loop( cond=_cond, 
                                                                  body=_levenberg_marquardt_optimization_step, 
                                                                  loop_vars=initial, 
                                                                  parallel_iterations=1)

    return final_objective_value, final_variables


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
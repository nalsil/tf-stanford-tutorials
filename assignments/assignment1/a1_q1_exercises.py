"""
Simple TensorFlow exercises
You should thoroughly test your code
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # default value = 0  From http://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information

import tensorflow as tf
from functions import showOperation as showOp


# sess ​=​ tf​.InteractiveSession​()
# with tf.InteractiveSession() as sess:


###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform([], minval=-1, maxval=1)
y = tf.random_uniform([], minval=-1, maxval=1)
zero_fn = tf.constant(0.0)

#x = tf.constant(1.5)
#y = tf.constant(1.5)

print('========================')
showOp(x)
showOp(y)

def f1():
    return tf.add(x, y)

def f2():
    return tf.cond( x > y, lambda: tf.subtract(x, y),  lambda: zero_fn)

out = tf.cond( x < y, lambda: f1(), lambda: f2() )
showOp(out)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################


x = tf.constant([[0, -2, -1], [0, 1, 2]], tf.float32)
y = tf.zeros_like(x)
z = tf.equal(x, y)

print('========================')
showOp(x)
showOp(y)
showOp(z)

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

print('========================')

x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
30.97266006,  26.67541885,  38.08450317,  20.74983215,
34.94445419,  34.45999146,  29.06485367,  36.01657104,
27.88236427,  20.56035233,  30.20379066,  29.51215172,
33.71149445,  28.59134293,  36.05556488,  28.66994858],
tf.float32)

y = tf.where( tf.greater(x, 30) )
showOp(y)

z  = tf.gather(x, tf.reshape( y , [1,-1]))
showOp(z)


###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

print('========================')

diag_range = tf.range(start=1, limit=7)
diag = tf.diag(diag_range)
showOp(diag)

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

print('========================')
x = tf.random_uniform([10, 10])
showOp(x)

detX = tf.matrix_determinant(x)
showOp(detX)

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

print('========================')
x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
y, idx = tf.unique(x)
showOp(y)
showOp(idx)

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

print('========================')
x = tf.random_normal([300])
y = tf.random_normal([300])

diff = tf.subtract(x, y)
avg = tf.reduce_mean(diff)
result = tf.where(avg < 0, tf.reduce_mean(tf.square(diff)), tf.reduce_sum(tf.abs(diff)))

showOp(result)

import tensorflow as tf
from inception_tensorflow import *
trunc_normal = lambda stddev : tf.truncated_normal_initializer(0.0, stddev)









neverusd_varaiable = tf.placeholder(tf.int64, shape= [1,5], name="never")
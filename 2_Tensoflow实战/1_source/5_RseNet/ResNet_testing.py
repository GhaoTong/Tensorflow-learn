import collections
import tensorflow as tf
import resnet_utils
import resnet_v2
import argparse
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import math
import time
from datetime import datetime

from resnet_v2 import slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

FLAGS = None


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                    (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
            (datetime.now(), info_string, FLAGS.num_batches, mn, sd))

def main (_):
    
    height, width = 224,224
    inputs = tf.random_uniform((FLAGS.batch_size,height,width,3))
    with slim.arg_scope(resnet_arg_scope()):
        net, end_point = resnet_v2.resnet_v2_152(inputs, 1000)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    time_tensorflow_run(sess, net, "Forward")

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=100,
        help='Number of batches to run.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print (FLAGS, unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




neverused = tf.placeholder(tf.float32, shape= [1,1])
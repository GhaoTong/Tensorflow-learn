import tensorflow as tf
from inception_tensorflow import inception_utils
from inception_tensorflow import inception_v3
import time
from datetime import datetime
import math
import argparse
from six.moves import xrange  # pylint: disable=redefined-builtin
import sys
trunc_normal = lambda stddev : tf.truncated_normal_initializer(0.0, stddev)

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
    height, width = 299, 299
    inputs = tf.random_uniform((FLAGS.batch_size, height, width, 3))
    with inception_v3.slim.arg_scope(inception_utils.inception_arg_scope()):
        logits, end_point = inception_v3.inception_v3(inputs,is_training= False)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    time_tensorflow_run(sess, logits, "Forward")
    
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




neverusd_varaiable = tf.placeholder(tf.int64, shape= [1,5], name="never")
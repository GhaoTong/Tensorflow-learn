import collections
import tensorflow as tf
import resnet_utils



class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block.'

def subsample (inputs, factor, scope = None):
    if factor == 1 :
        return inputs
    else :
        return slim.max_pool2d(inputs, [1,1],stride = factor, scope=scope)

def conv2d_same(inputs, num_outputs, kenal_size, strides, scope= None):
    if strides  == 1 :
        return slim.conv2d(inputs, num_outputs, kenal_size, strides=1,padding='SAME',scope= scope)
    else :
        pad_total = kenal_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0,0],[pad_beg,pad_end],
                                [pad_beg,pad_end],[0,0]])

@slim.add_arg_scope
def stack_blocke_dense(net, blocks, outputs_collections= None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block',[net]) as sc :
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth = unit_depth,
                                        depth_bottleneck = unit_depth_bottleneck,
                                        stride = unit_stride)
                net = slim.unitls.collect_named_outputs(outputs_collections, sc.name, net)
    return net

def resnet_arg_scope(is_training =True,
                    weight_decay = 0.0001,
                    batch_norm_dacay = 0.997,
                    batch_norm_epsilon = 1e-5,
                    batch_norm_scale = True):
    batch_norm_params = {
        'is_training':is_training,
        'decay': batch_norm_dacay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        # The following implies padding='SAME' for pool1, which makes feature
        # alignment easier for dense prediction tasks. This is also used in
        # https://github.com/facebook/fb.resnet.torch. However the accompanying
        # code of 'Deep Residual Learning for Image Recognition' uses
        # padding='VALID' for pool1. You can switch to that choice by setting
        # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc




neverused = tf.placeholder(tf.float32, shape= [1,1])
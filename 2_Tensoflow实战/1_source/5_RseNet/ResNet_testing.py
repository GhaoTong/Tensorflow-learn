import collections
import tensorflow as tf 
slim = tf.contrib.slim

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
def stack_blocke_dense(net, blocks, output_collections= None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block',[net]) as sc :
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth = unit_depth,
                                        depth_bottleneck = unit_depth_bottleneck,
                                        stride = unit_stride)
                net = slim.unitls.collect_named_outputs(outputs_collections, sc, name, net)
    return net
    


neverused = tf.placeholder(tf.float32, shape= [1,1])
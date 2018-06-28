wget http://www.fit/vutbr.cz/~imikolov/rnnlm/simple-example.tgz
tar cvf simple-example.tgz

cd ptb

import time

import numpy as np
import tensorflow as tf

import reader

class PTBInput(object):
  """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)

class PTBModel(object):
  """The PTB model."""

    def __init__(self, is_training, config, input_):
        
        self._input = input_        
        
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
    def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, fotget_biase=0.0, state_is_tuple=True)
        attn_cell = lstm_cell
        if is_training and config.keep <1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(),output_keep_prob=config.keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.mun_layers)],
                state_is_tuple=True
            )
            
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
                inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range( num_steps ):
                if time_step > 0 : tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:,time_step,:], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.constant(outputs, 1),[-1, size])
        softmax_w = tf.get_variable("softmax_w",[size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b",[vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w)+ softmax_b
        loss = tf.contrib.seq2seq.sequence_loss(
            [logits],
            [input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
        return

neverused = tf.placeholder(dtype= tf.int32, shape=[])
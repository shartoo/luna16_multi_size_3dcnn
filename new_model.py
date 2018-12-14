import tensorflow as tf
import numpy as np

DTYPE = tf.float32
_keep_prob = 0.8
def weight_variable(name,shape):
    return tf.get_variable(name,shape,DTYPE,tf.truncated_normal_initializer(stddev=0.001))

def bias_variable(name,shape):
    return tf.get_variable(name,shape,DTYPE,tf.constant(0.1,dtype=DTYPE))

def arch1(inputs):
    '''
        network structure for cube shape 20x20x6  ，cover 58%  nodule
        :param inputs:
        :return:
        '''
    in_filters = 1
    pre_layer = inputs
    with tf.name_scope("arch-1") as scope:
        # shape of input for arch-1 is cube 20x20x6
        with tf.variable_scope("conv_1") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight",[3,5,5,in_filters,out_filters])
            # the output size is batch_size x 6x20x20x64 ([batch_size,in_deep,width,height,output_deep])
            conv = tf.nn.conv3d(pre_layer,kernel,strides=[1,1,1,1,1],padding='SAME')
            bias = bias_variable("biases",[out_filters])
            bias = tf.nn.bias_add(conv,bias)
            conv1 = tf.nn.relu(bias,name=con_scope.name)

            pre_layer = conv1
            in_filters = out_filters

        pre_layer = tf.nn.dropout(pre_layer,keep_prob=_keep_prob)
        # pooling don't change feature shape at all
        pool1 = tf.nn.max_pool3d(pre_layer,ksize=[1,1,1,1,1],strides=[1,1,1,1,1],padding='SAME')
        pre_layer= pool1

        with tf.variable_scope("conv_2") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight", [3,5,5, in_filters, out_filters])
            conv = tf.nn.conv3d(pre_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            bias = bias_variable("biases", [out_filters])
            bias = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(bias, name=con_scope.name)

            pre_layer = conv2
            in_filters = out_filters
        pre_layer = tf.nn.dropout(pre_layer, keep_prob=_keep_prob)

        with tf.variable_scope("conv_3") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight", [1,5,5, in_filters, out_filters])
            conv = tf.nn.conv3d(pre_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            bias = bias_variable("biases", [out_filters])
            bias = tf.nn.bias_add(conv, bias)
            conv3 = tf.nn.relu(bias, name=con_scope.name)

            pre_layer = conv3
            in_filters = out_filters
        pre_layer = tf.nn.dropout(pre_layer, keep_prob=_keep_prob)
        # output shape of all above is 6x20x20x64,maxpooling kernel is 1x1x1, and convolutional padding are same,no shape decrease
        out_conv3 = tf.reshape(pre_layer, [-1, 6 * 20 * 20 * 64])
        w_fc1 = weight_variable([6 * 20 * 20 * 64, 150],name='w_fc1')
        out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3, w_fc1), bias_variable([150],name="fc1_biases")))
        out_fc1 = tf.nn.dropout(out_fc1, keep_prob=_keep_prob)

        w_fc2 = weight_variable([150,2],name="w_fc2")
        out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1,w_fc2), bias_variable([2], name="fc2_biases")))
        out_fc2 = tf.nn.dropout(out_fc2, keep_prob=_keep_prob)
        return out_fc2

def arch2(inputs):
    '''
        network structure for cube shape 30x30x10 ，cover 85%  nodule
        :param inputs:
        :return:
        '''
    in_filters = 1
    pre_layer = inputs
    with tf.name_scope("arch-2") as scope:
        # shape of input for arch-1 is cube 30x30x10
        with tf.variable_scope("conv_1") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight",[3,5,5,in_filters,out_filters])
            # the output size is batch_size x 6x20x20x64 ([batch_size,in_deep,width,height,output_deep])
            conv = tf.nn.conv3d(pre_layer,kernel,strides=[1,1,1,1,1],padding='SAME')
            bias = bias_variable("biases",[out_filters])
            bias = tf.nn.bias_add(conv,bias)
            conv1 = tf.nn.relu(bias,name=con_scope.name)

            pre_layer = conv1
            in_filters = out_filters

        pre_layer = tf.nn.dropout(pre_layer,keep_prob=_keep_prob)
        # pooling make shape into  10x15x15x64
        pool1 = tf.nn.max_pool3d(pre_layer,ksize=[1,1,2,2,1],strides=[1,1,1,1,1],padding='SAME')
        pre_layer= pool1

        with tf.variable_scope("conv_2") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight", [3,5,5, in_filters, out_filters])
            conv = tf.nn.conv3d(pre_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            bias = bias_variable("biases", [out_filters])
            bias = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(bias, name=con_scope.name)

            pre_layer = conv2
            in_filters = out_filters
        pre_layer = tf.nn.dropout(pre_layer, keep_prob=_keep_prob)

        with tf.variable_scope("conv_3") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight", [3,5,5, in_filters, out_filters])
            conv = tf.nn.conv3d(pre_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            bias = bias_variable("biases", [out_filters])
            bias = tf.nn.bias_add(conv, bias)
            conv3 = tf.nn.relu(bias, name=con_scope.name)

            pre_layer = conv3
            in_filters = out_filters
        pre_layer = tf.nn.dropout(pre_layer, keep_prob=_keep_prob)
        # output shape of all above is 10x15x15x64, convolutional padding are SAME
        out_conv3 = tf.reshape(pre_layer, [-1, 10 * 15 * 15 * 64])
        w_fc1 = weight_variable([10 * 15 * 15 * 64, 250],name='w_fc1')
        out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3, w_fc1), bias_variable([250],name="fc1_biases")))
        out_fc1 = tf.nn.dropout(out_fc1, keep_prob=_keep_prob)

        w_fc2 = weight_variable([250,2],name="w_fc2")
        out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1,w_fc2), bias_variable([2], name="fc2_biases")))
        out_fc2 = tf.nn.dropout(out_fc2, keep_prob=_keep_prob)
        return out_fc2


def arch3(inputs):
    '''
    network structure for cube shape 40x40x26 ，cover 99%  nodule
    :param inputs:
    :return:
    '''
    in_filters = 1
    pre_layer = inputs
    with tf.name_scope("arch-3") as scope:
        # shape of input for arch-1 is cube 40x40x26
        with tf.variable_scope("conv_1") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight",[3,5,5,in_filters,out_filters])
            # the output size is batch_size x 6x20x20x64 ([batch_size,in_deep,width,height,output_deep])
            conv = tf.nn.conv3d(pre_layer,kernel,strides=[1,1,1,1,1],padding='SAME')
            bias = bias_variable("biases",[out_filters])
            bias = tf.nn.bias_add(conv,bias)
            conv1 = tf.nn.relu(bias,name=con_scope.name)

            pre_layer = conv1
            in_filters = out_filters

        pre_layer = tf.nn.dropout(pre_layer,keep_prob=_keep_prob)
        # pooling make shape into  13x20x20x64
        pool1 = tf.nn.max_pool3d(pre_layer,ksize=[1,2,2,2,1],strides=[1,1,1,1,1],padding='SAME')
        pre_layer= pool1

        with tf.variable_scope("conv_2") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight", [3,5,5, in_filters, out_filters])
            conv = tf.nn.conv3d(pre_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            bias = bias_variable("biases", [out_filters])
            bias = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(bias, name=con_scope.name)

            pre_layer = conv2
            in_filters = out_filters
        pre_layer = tf.nn.dropout(pre_layer, keep_prob=_keep_prob)

        with tf.variable_scope("conv_3") as con_scope:
            out_filters = 64
            kernel = weight_variable("weight", [3,5,5, in_filters, out_filters])
            conv = tf.nn.conv3d(pre_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            bias = bias_variable("biases", [out_filters])
            bias = tf.nn.bias_add(conv, bias)
            conv3 = tf.nn.relu(bias, name=con_scope.name)

            pre_layer = conv3
            in_filters = out_filters
        pre_layer = tf.nn.dropout(pre_layer, keep_prob=_keep_prob)
        # output shape of all above is 13x20x20x64, convolutional padding are SAME
        out_conv3 = tf.reshape(pre_layer, [-1, 13 * 20 * 20 * 64])
        w_fc1 = weight_variable([13 * 20 * 20 * 64, 250],name='w_fc1')
        out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3, w_fc1), bias_variable([250],name="fc1_biases")))
        out_fc1 = tf.nn.dropout(out_fc1, keep_prob=_keep_prob)

        w_fc2 = weight_variable([250,2],name="w_fc2")
        out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1,w_fc2), bias_variable([2], name="fc2_biases")))
        out_fc2 = tf.nn.dropout(out_fc2, keep_prob=_keep_prob)
        return out_fc2




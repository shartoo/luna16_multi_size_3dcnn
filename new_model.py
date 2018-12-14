import tensorflow as tf
import numpy as np
import time
from data_prepare import get_all_filename,get_train_batch,get_test_batch
import random

DTYPE = tf.float32
learning_rate = 0.3
epoch = 1000
cubic_shape = [[6, 20, 20], [10, 30, 30], [26, 40, 40]]
alpha1= 0.3
alpha2 = 0.4
alpha3 = 0.3

gpu_options1 = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options1))

gpu_options2 = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options2))

gpu_options3 = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess3 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options3))

def weight_variable(name,shape):
    return tf.get_variable(name,shape,DTYPE,tf.truncated_normal_initializer(stddev=0.001))

def bias_variable(name,shape):
    return tf.get_variable(name,shape,DTYPE,tf.constant(0.1,dtype=DTYPE))

def arch1(inputs,_keep_prob):
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

def arch2(inputs,_keep_prob):
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


def arch3(inputs,_keep_prob):
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


def train_model(arch_index,npy_path,test_path,batch_size = 32):

        highest_acc = 0.0
        highest_iterator = 1
        all_filenames = get_all_filename(npy_path,cubic_shape[arch_index][1])
        print("file size is :\t",len(all_filenames))
        # how many time should one epoch should loop to feed all data
        times = int(len(all_filenames) / batch_size)
        if (len(all_filenames) % batch_size) != 0:
            times = times + 1

        # keep_prob used for dropout
        keep_prob = tf.placeholder(tf.float32)
        # take placeholder as input
        x = tf.placeholder(tf.float32, [None, cubic_shape[arch_index][0], cubic_shape[arch_index][1], cubic_shape[arch_index][2]])
        x_image = tf.reshape(x, [-1, cubic_shape[arch_index][0], cubic_shape[arch_index][1], cubic_shape[arch_index][2], 1])
        if arch_index == 1:
            net = arch1(x_image)
        elif arch_index == 2:
            net = arch2(x_image)
        elif arch_index == 3:
            net = arch3(x_image)
        else:
            print("model architecture index must be 1 or 2 or 3,current is %s which is not supported"%(str(arch_index)))
            return

        saver = tf.train.Saver()  # default to save all variable,save mode or restore from path
        # softmax layer
        real_label = tf.placeholder(tf.float32, [None, 2])
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=real_label))
        net_loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss)
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(real_label, 1))
        accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('./arch-%d-tensorboard/'%(arch_index), sess.graph)
            # loop epoches
            for i in range(epoch):
                epoch_start =time.time()
                #  the data will be shuffled by every epoch
                random.shuffle(all_filenames)
                for t in range(times):
                    batch_files = all_filenames[t*batch_size:(t+1)*batch_size]
                    batch_data, batch_label = get_train_batch(batch_files)
                    feed_dict = {x: batch_data, real_label: batch_label,keep_prob:0.8}
                    _,summary = sess.run([train_step, merged],feed_dict =feed_dict)
                    train_writer.add_summary(summary, i)
                    saver.save(sess, './arch-%d-ckpt/arch-%d'%(arch_index,arch_index), global_step=i + 1)

                epoch_end = time.time()
                test_batch,test_label = get_test_batch(test_path)
                print("test batch data:\t",test_batch)
                print("test batch label:\t", test_label)
                test_dict = {x: test_batch, real_label: test_label, keep_prob:1.0}
                acc_test,loss = sess.run([accruacy,net_loss],feed_dict=test_dict)
                print('accuracy  is %f' % acc_test)
                print("loss is ", loss)
                print(" epoch %d time consumed %f seconds"%(i,(epoch_end-epoch_start)))

            print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))


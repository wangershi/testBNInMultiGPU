# -*- coding:utf-8 -*-
'''
	Program:
		test Batch Normalization in multi GPU
	Release:
		2019/06/15	ZhangDao	First release
'''
import tensorflow as tf
import numpy as np
import os

def testBNInSingleGPU():
	'''
		Test Batch Normalization in single GPU
		Args:
			None
		Returns:
			None
	'''
	'''
		compute GPU number and memory
	'''
	memoryList = list(map(int, os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total\
		| awk '{print $3}'").readlines()))
	GPUNumber = len(memoryList)
	GPUMemorySize = memoryList[0]

	'''
		configment of the TensorFlow
	'''
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allocator_type = 'BFC'

	memoryLimited = 200	# memory for CRNN to train
	config.gpu_options.per_process_gpu_memory_fraction = memoryLimited / GPUMemorySize

	'''
		network
	'''
	x = tf.placeholder(tf.float32, shape=[3], name='data')
	y = tf.layers.batch_normalization(x, momentum=0.9, training=True)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	'''
		train
	'''
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		inputArray = np.array((1., 1., 1.))
		feedDict = {
			x:	inputArray
		}
		fetchList = [
			y,\
			update_ops
		]
		outputArray, _ = sess.run(fetches=fetchList, feed_dict=feedDict)

		print (f"outputArray = {outputArray}")

		print ("\nAfter normalization:")
		movingMean = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_mean:0")
		print ("moving mean = %s" % sess.run(movingMean))
		movingVariance = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_variance:0")
		print ("moving variance = %s" % sess.run(movingVariance))

def testBNInMultiGPU1():
	'''
		Test Batch Normalization in multi GPU
		Args:
			None
		Returns:
			None
	'''
	'''
		compute GPU number and memory
	'''
	memoryList = list(map(int, os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total\
		| awk '{print $3}'").readlines()))
	GPUNumber = len(memoryList)
	GPUMemorySize = memoryList[0]

	'''
		configment of the TensorFlow
	'''
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allocator_type = 'BFC'

	memoryLimited = 200	# memory for CRNN to train
	config.gpu_options.per_process_gpu_memory_fraction = memoryLimited / GPUMemorySize

	'''
		network
	'''
	numberGPU = 2
	with tf.variable_scope(tf.get_variable_scope()):
		for itemGPU in range(numberGPU):
			with tf.device("/gpu:%d" % itemGPU):
				with tf.name_scope("tower_%d" % itemGPU):
					x = tf.placeholder(tf.float32, shape=[3], name='data')
					y = tf.layers.batch_normalization(x, momentum=0.9, training=True, reuse=tf.AUTO_REUSE)
					tf.get_variable_scope().reuse_variables()

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	'''
		train
	'''
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		inputArray = np.array((1., 1., 1.))
		feedDict = {}
		for item in range(numberGPU):
			feedDict[tf.get_default_graph().get_tensor_by_name("tower_%s/data:0" % item)] = inputArray
		fetchList = [
			y,\
			update_ops
		]
		outputArray, _ = sess.run(fetches=fetchList, feed_dict=feedDict)
		print (f"outputArray = {outputArray}")

		print ("\nAfter normalization:")
		movingMean = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_mean:0")
		print ("moving mean = %s" % sess.run(movingMean))
		movingVariance = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_variance:0")
		print ("moving variance = %s" % sess.run(movingVariance))
		
def testBNInMultiGPU2():
	'''
		Test Batch Normalization in multi GPU
		Args:
			None
		Returns:
			None
	'''
	'''
		compute GPU number and memory
	'''
	memoryList = list(map(int, os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total\
		| awk '{print $3}'").readlines()))
	GPUNumber = len(memoryList)
	GPUMemorySize = memoryList[0]

	'''
		configment of the TensorFlow
	'''
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allocator_type = 'BFC'

	memoryLimited = 200	# memory for CRNN to train
	config.gpu_options.per_process_gpu_memory_fraction = memoryLimited / GPUMemorySize

	'''
		network
	'''
	numberGPU = 2
	with tf.variable_scope(tf.get_variable_scope()):
		for itemGPU in range(numberGPU):
			with tf.device("/gpu:%d" % itemGPU):
				with tf.name_scope("tower_%d" % itemGPU):
					x = tf.placeholder(tf.float32, shape=[3], name='data')
					y = tf.layers.batch_normalization(x, momentum=0.9, training=True, reuse=tf.AUTO_REUSE)
					tf.get_variable_scope().reuse_variables()

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	'''
		train
	'''
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		feedDict = {}
		for item in range(numberGPU):
			feedDict[tf.get_default_graph().get_tensor_by_name("tower_%s/data:0" % item)]\
				= np.ones((3))*(item+1)
		fetchList = [
			y,\
			update_ops
		]
		outputArray, _ = sess.run(fetches=fetchList, feed_dict=feedDict)
		print (f"outputArray = {outputArray}")

		print ("\nAfter normalization:")
		movingMean = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_mean:0")
		print ("moving mean = %s" % sess.run(movingMean))
		movingVariance = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_variance:0")
		print ("moving variance = %s" % sess.run(movingVariance))
		
from tensorflow.contrib.nccl.ops import gen_nccl_ops
from tensorflow.contrib.framework import add_model_variable

def sync_batch_norm(inputs,
					decay=0.999,
					axis=-1,
					epsilon=0.001,
					activation_fn=None,
					updates_collections=tf.GraphKeys.UPDATE_OPS,
					is_training=True,
					reuse=None,
					variables_collections=None,
					trainable=True,
					scope=None,
					num_dev=1):
	'''
		num_dev is how many gpus you use.
		this function is from https://github.com/jianlong-yuan/syncbn-tensorflow/blob/master/syncbn.py
	'''
	# shape of inputs is [batch, height, width, depth]
	num_outputs = inputs.get_shape().as_list()[-1]
	# print (f"num_outputs = {num_outputs}")	# 3

	if scope is None:
		scope = 'batch_normalization'

	with tf.variable_scope(scope, reuse=reuse):
		# initializer, gamma and beta is trainable, moving_mean and moving_var is not
		gamma = tf.get_variable(name='gamma', shape=[num_outputs], dtype=tf.float32,
			initializer=tf.constant_initializer(1.0), trainable=trainable,
			collections=variables_collections)

		beta  = tf.get_variable(name='beta', shape=[num_outputs], dtype=tf.float32,
			initializer=tf.constant_initializer(0.0), trainable=trainable,
			collections=variables_collections)

		moving_mean = tf.get_variable(name='moving_mean', shape=[num_outputs], dtype=tf.float32,
			initializer=tf.constant_initializer(0.0), trainable=False,
			collections=variables_collections)

		moving_var = tf.get_variable(name='moving_variance', shape=[num_outputs], dtype=tf.float32,
			initializer=tf.constant_initializer(1.0), trainable=False,
			collections=variables_collections)

		# is_training and trainable is logical and
		# this is same with [math_ops.logical_and())]
		# (https://github.com/tensorflow/tensorflow/blob/
		# 508f76b1d9925304cedd56d51480ec380636cb82/tensorflow/
		# python/keras/layers/normalization.py#L621)
		if is_training and trainable:
			# only one GPU
			if num_dev == 1:
				mean, var = tf.nn.moments(inputs, axes=axis)
			# multi GPUs
			else:
				# avarage moving_mean and moving_var in multi GPUs
				shared_name = tf.get_variable_scope().name
				batch_mean = tf.reduce_mean(inputs, axis=axis)
				batch_mean = gen_nccl_ops.nccl_all_reduce(
					input=batch_mean,
					reduction='sum',
					num_devices=num_dev,
					shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
				batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=axis)
				batch_mean_square = gen_nccl_ops.nccl_all_reduce(
					input=batch_mean_square,
					reduction='sum',
					num_devices=num_dev,
					shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
				mean = batch_mean
				var = batch_mean_square - tf.square(batch_mean)
			outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

			# print (outputs.device)	# /device:GPU:1

			# those code block is executed in every GPUs
			# just assign moving_mean and moving_var in GPU:0
			if int(outputs.device[-1]) == 0:
				update_moving_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
				update_moving_var_op  = tf.assign(moving_var,  moving_var  * decay + var  * (1 - decay))
				add_model_variable(moving_mean)
				add_model_variable(moving_var)

				if updates_collections is None:
					with tf.control_dependencies([update_moving_mean_op, update_moving_var_op]):
						outputs = tf.identity(outputs)
				else:
					tf.add_to_collections(updates_collections, update_moving_mean_op)
					tf.add_to_collections(updates_collections, update_moving_var_op)
					outputs = tf.identity(outputs)
			else:
				outputs = tf.identity(outputs)
		else:
			outputs, _, _ = tf.nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean, variance=moving_var, epsilon=epsilon, is_training=False)

		if activation_fn is not None:
			outputs = activation_fn(outputs)

		return outputs

def testBNInMultiGPU3():
	'''
		Test Batch Normalization in multi GPU
		Test function sync_batch_norm()
		Args:
			None
		Returns:
			None
	'''
	'''
		compute GPU number and memory
	'''
	memoryList = list(map(int, os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total\
		| awk '{print $3}'").readlines()))
	GPUNumber = len(memoryList)
	GPUMemorySize = memoryList[0]

	'''
		configment of the TensorFlow
	'''
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allocator_type = 'BFC'

	memoryLimited = 200	# memory for CRNN to train
	config.gpu_options.per_process_gpu_memory_fraction = memoryLimited / GPUMemorySize

	'''
		network
	'''
	numberGPU = 2
	with tf.variable_scope(tf.get_variable_scope()):
		for itemGPU in range(numberGPU):
			with tf.device("/gpu:%d" % itemGPU):
				with tf.name_scope("tower_%d" % itemGPU):
					x = tf.placeholder(tf.float32, shape=[3], name='data')
					locals()['y%s' % itemGPU] = sync_batch_norm(x, decay=0.9,\
						is_training=True, reuse=tf.AUTO_REUSE, num_dev=numberGPU)
					tf.get_variable_scope().reuse_variables()

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	print (tf.global_variables())

	'''
		train
	'''
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		feedDict = {}
		fetchList = []
		for item in range(numberGPU):
			feedDict[tf.get_default_graph().get_tensor_by_name("tower_%s/data:0" % item)]\
				= np.ones((3))*(item+1)
			fetchList.append(locals()['y%s' % item])
		fetchList.append(update_ops)

		outputTuple = (locals()['outputArray%s' % item] for item in range(numberGPU+1))
		outputTuple = sess.run(fetches=fetchList, feed_dict=feedDict)
		print (f"outputTuple = {outputTuple}")

		print ("\nAfter normalization:")
		movingMean = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_mean:0")
		print ("moving mean = %s" % sess.run(movingMean))
		movingVariance = tf.get_default_graph().get_tensor_by_name("batch_normalization/moving_variance:0")
		print ("moving variance = %s" % sess.run(movingVariance))

if __name__ == "__main__":
	# testBNInSingleGPU()
	# testBNInMultiGPU1()
	# testBNInMultiGPU2()
	testBNInMultiGPU3()

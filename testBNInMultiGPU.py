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

if __name__ == "__main__":
	# testBNInSingleGPU()
	# testBNInMultiGPU1()
	testBNInMultiGPU2()

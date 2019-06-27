# -*- coding:utf-8 -*-
'''
	Program:
		sync Batch Normalization in multi GPU
	Release:
		2019/06/27	ZhangDao	First release
'''
import tensorflow as tf
import numpy as np
import os, re

# TF version ls lower/equal with tf.1.12.0
# this code is from [batch_norm.py](https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/batch_norm.py)
if tuple(map(int, tf.__version__.split('.')[:2])) <= (1, 12):
	try:
		from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
	except Exception:
		pass
	else:
		_validate_and_load_nccl_so()
	from tensorflow.contrib.nccl.ops import gen_nccl_ops
else:
    from tensorflow.python.ops import gen_nccl_ops

from tensorflow.contrib.framework import add_model_variable

def syncBatchNorm(inputs, 
					axis=-1,
					momentum=0.99,
					epsilon=0.001,
					updates_collections=tf.GraphKeys.UPDATE_OPS,
					reuse=None,
					variables_collections=None,
					training=False,
					trainable=True,
					name=None,
					GPUNumber=1):
	'''
		this function is from https://github.com/jianlong-yuan/syncbn-tensorflow/blob/master/syncbn.py
	'''
	shapeList = inputs.get_shape().as_list()
	num_outputs = shapeList[axis]
	# print (f"num_outputs = {num_outputs}")	# 512
	axes = [i for i in range(len(shapeList))]
	# when the dimension is 1, axes = [], this also run well!
	del axes[axis]
	# print (f"axes = {axes}")	# [0, 1, 2]

	if name is None:
		name = 'batch_normalization'

	with tf.variable_scope(name, reuse=reuse) as scope:
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

		def branchTrue():
			'''
				update the batch mean and batch variance
			'''
			# only one GPU
			if GPUNumber == 1:
				batch_mean = tf.reduce_mean(inputs, axis=axes, name="batch_mean")
				batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=axes)
			# multi GPUs
			else:
				# avarage moving_mean and moving_var in multi GPUs
				shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
				batch_mean = tf.reduce_mean(inputs, axis=axes)

				# Utilize NCCL
				batch_mean = gen_nccl_ops.nccl_all_reduce(
					input=batch_mean,
					reduction='sum',
					num_devices=GPUNumber,
					shared_name=shared_name + '_NCCL_mean') * (1.0 / GPUNumber)
				batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=axes)
				batch_mean_square = gen_nccl_ops.nccl_all_reduce(
					input=batch_mean_square,
					reduction='sum',
					num_devices=GPUNumber,
					shared_name=shared_name + '_NCCL_mean_square') * (1.0 / GPUNumber)
				
			batch_var = batch_mean_square - tf.square(batch_mean)

			outputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

			return outputs, batch_mean, batch_var

		def branchFalse():
			'''
				the same with moving_mean and moving_var
			'''
			outputs = tf.nn.batch_normalization(inputs, moving_mean, moving_var, beta, gamma, epsilon)
			
			# use the default tensor, this code will not update moving_mean and moving_var
			# for batch_mean+(moving_mean-batch_mean)*momentum = moving_mean
			# is batch_mean == moving_mean
			with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
				batch_mean = tf.get_variable("moving_mean")
				batch_var = tf.get_variable("moving_variance")
			return outputs, batch_mean, batch_var

		outputs, batch_mean, batch_var = tf.cond(tf.math.logical_and(training, trainable), branchTrue, branchFalse)

		# those code block is executed in every GPUs
		# just assign moving_mean and moving_var in GPU:0
		if int(outputs.device[-1]) == 0:
			update_moving_mean_op = tf.assign(moving_mean, batch_mean+(moving_mean-batch_mean)*momentum)
			update_moving_var_op  = tf.assign(moving_var,  batch_var+(moving_var-batch_var)*momentum)
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

		return outputs

def testBNInMultiGPU():
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
	dimension = [3]
	ifTraining = True
	with tf.variable_scope(tf.get_variable_scope()):
		for itemGPU in range(numberGPU):
			with tf.device("/gpu:%d" % itemGPU):
				with tf.name_scope("tower_%d" % itemGPU):
					x = tf.placeholder(tf.float32, shape=dimension, name='data')
					training = tf.placeholder(tf.bool, shape=(), name='training')
					locals()['y%s' % itemGPU] = syncBatchNorm(x, momentum=0.9,\
						training=training, reuse=tf.AUTO_REUSE, GPUNumber=numberGPU)
					tf.get_variable_scope().reuse_variables()

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# print (update_ops)	# [<tf.Tensor 'tower_0/batch_normalization/Assign:0' shape=(3,) dtype=float32_ref>, <tf.Tensor 'tower_0/batch_normalization/Assign_1:0' shape=(3,) dtype=float32_ref>]

	# print ([n.name for n in tf.get_default_graph().as_graph_def().node])

	'''
		train
	'''
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		feedDict = {}
		fetchList = []
		for item in range(numberGPU):
			feedDict[tf.get_default_graph().get_tensor_by_name("tower_%s/data:0" % item)]\
				= np.ones(dimension)*(item+1)
			feedDict[tf.get_default_graph().get_tensor_by_name("tower_%s/training:0" % item)]\
				= ifTraining
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
	testBNInMultiGPU()

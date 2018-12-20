#===========================#
#  Read the TFRecords file  #
#===========================#
'''
Read data from the TFRecords file:
	- We load the data from the train data in batchs of an arbitrary size 
	- and plot images of the 5 batchs. 
	- We also check the label of each image. 

To read from files in tensorflow, you need to do the following steps:
	- Create a list of filenames: 
		In our case we only have a single file data_path = 'train.tfrecords'. Therefore, our list is gonna be like this: [data_path]
	- Create a queue to hold filenames: 
		To do so, we use tf.train.string_input_producer function which hold filenames in a FIFO queue. 
		it gets the list of filnames. 
		It also has some optional arguments including:  
			num_epochs, which indicates the number of epoch you want to to load the data and 
			shuffle, which indicates whether to suffle the filenames in the list or not. It is set to True by default.
	- Define a reader: 
		For files of TFRecords we need to define a TFRecordReader with reader = tf.TFRecordReader().
		Now, the reader returns the next record using: reader.read(filename_queue)
	- Define a decoder: 
		A decoder is needed to decode the record read by the reader. 
		In case of using TFRecords files the decoder should be tf.parse_single_example. 
		it takes a serialized Example and a dictionary which maps feature keys to FixedLenFeature or VarLenFeature values and 
		returns a dictionary which maps feature keys to Tensor values: 
				features = tf.parse_single_example(serialized_example, features=feature)
	- Convert the data from string back to the numbers: 
		tf.decode_raw(bytes, out_type) takes a Tensor of type string and convert it to typeout_type. 
		However, for labels which have not been converted to string, we just need to cast them using tf.cast(x, dtype)
	- Reshape data into its original shape: 
		You should reshape the data (image) into it's original shape before serialization using image = tf.reshape(image, [224, 224, 3])
	- Preprocessing: 
		if you want to do any preprocessing you should do it now.
	- Batching: 
		Another queue is needed to create batches from the examples. 
		You can create the batch queue using tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10) 
		where capacity is the maximum size of queue, min_after_dequeue is the minimum size of queue after dequeue, and num_threads is the number of threads enqueuing examples. 
		Using more than one thread, it comes up with a faster reading. The first argument in a list of tensors which you want to create batches from.
'''
#=========================#
#  Read a TFRecords file  #
#=========================#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = 'mouth_train.tfrecords'     # address to save the hdf5 file

with tf.Session() as sess:
	# Define the features you expect in the TFRecord by using tf.FixedLenFeature and 
	# tf.VarLenFeature, depending on what has been defined during the defination of tf. train.Example
	feature = {
			'train/image':tf.FixedLenFeature([],tf.string),
			'train/label':tf.FixedLenFeature([],tf.int64)
	}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path],num_epochs=1)

	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example,features = feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['train/image'],tf.float32)

	# Cast label data into int32
	label = tf.cast(features['train/label'],tf.int32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [60,60,3])

	# Any preprocessing here ...

	# Creates batches by randomly shuffling tensors
	images, labels = tf.train.shuffle_batch([image,label],batch_size = 10, capacity = 30, num_threads = 1, min_after_dequeue = 10)

	'''
	- Initialize all global and local variables
	- Filing the example queue: 
		Some functions of tf.train such as tf.train.shuffle_batch add tf.train.QueueRunner objects to your graph. 
		Each of these objects hold a list of enqueue op for a queue to run in a thread. 
		Therefore, to fill a queue you need to call tf.train.start_queue_runners which starts threades for all the queue runners in the graph. 
		However, to manage these threads you need a tf.train.Coordinator to terminate the threads at the proper time.
	- Everything is ready. Now you can read a batch and plot all batch images and labels. Do not forget to stop the threads (by stopping the cordinator) when you are done with your reading process.
	'''

	# Initialize all global and local variables
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)

	# Create a coordinator and run all QueueRunner objects
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	for batch_index in range(5):
		img, lbl = sess.run([images, labels])

		img = img.astype(np.uint8)

		for j in range(6):
			plt.subplot(2,3,j+1)
			plt.imshow(img[j,...])
			plt.title('smile' if lbl[j]==1 else 'nonsmile')

		plt.show()

	# Stop the threads
	coord.request_stop()

	# Wait for threads to stop
	coord.join(threads)
	sess.close()





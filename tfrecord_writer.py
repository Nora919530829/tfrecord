#================================#
#  List images and their labels  #
#================================#

''' 
	We give each nonsmile image a label = 0 and each smile image a label = 1. 
    List all images, give them proper labels, and then shuffle the data. 
    We also divide the data set into three train (%60), validation (%20), and test parts (%20).
'''

from random import shuffle
import os
import tensorflow as tf
import cv2
import numpy as np
import sys
'''
# 新建train，test，val文件夹，用来放图片数据
if not os.path.exists('./train'): 
	os.makedirs('./train')             
if not os.path.exists('./test'): 
	os.makedirs('./test') 
if not os.path.exists('./val'): 
	os.makedirs('./val')    
'''        

shuffle_data = True  # shuffle the addresses before saving
nonsmile_smile_train_path = 'mouth/data/*.jpg'

# read addresses and labels from the 'train' folder
addrs = os.listdir('data')
labels = [0 if 'neutral' in addr else 1 for addr in addrs]

# to shuffle data
if shuffle_data:
	c = list(zip(addrs,labels))
	shuffle(c)
	addrs,labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]

val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

print("train_labels: %d, including smile_num: %d, train_addrs type: %s" %(len(list(train_labels)),list(train_labels).count(1),type(train_addrs)))
print("val_labels: %d, including smile_num: %d" %(len(list(val_labels)),list(val_labels).count(1)))
print("test_labels: %d, including smile_num: %d" %(len(list(test_labels)),list(test_labels).count(1)))


#=====================#
#  create a TFRecord  #
#=====================#
'''
	First we need to load the image and convert it to the data type (float32 in this example) 
	in which we want to save the data into a TFRecords file. 
	Write a function which take an image address, load, resize, and return the image in proper data type
'''

#=============================#
#  A function to Load images  #
#=============================#
def load_image(addr):
	# read an image and resize to (60,60)
	# cv2 load images as BGR, convert it to RGB
	img = cv2.imread(addr)
	img = cv2.resize(img, (60,60), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.astype(np.float32)
	return img

'''
	Before we can store the data into a TFRecords file, we should stuff it in a protocol buffer called Example. 
	Then, we serialize the protocol buffer to a string and write it to a TFRecords file. 
		- Example protocol buffer contains Features. 
		- Feature is a protocol to describe the data and could have three types: bytes, float, and int64. 

	In summary, to store your data you need to follow these steps:		
		- Open a TFRecords file using tf.python_io.TFRecordWriter
		- Convert your data into the proper data type of the feature using tf.train.Int64List, tf.train.BytesList, or tf.train.FloatList
		- Create a feature using tf.train.Feature and pass the converted data to it
		- Create an Example protocol buffer using tf.train.Example and pass the feature to it
		- Serialize the Example to string using example.SerializeToString()
		- Write the serialized example to TFRecords file using writer.write
'''

#============================#
#  Convert data to features  #
#============================#
def _int64_feature(value):
	return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

#==========================================#
#  Write train data into a TFRecords file  #
#==========================================#
train_filename = 'mouth_train.tfrecords' 	# address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename) 

for i in range(len(train_addrs)):
	# Load the image
	img = load_image(os.path.join('./data',train_addrs[i]))
	label = train_labels[i]

	# create a feature
	feature = {
		'train/label':_int64_feature(label), 
		'train/image':_bytes_feature(tf.compat.as_bytes(img.tostring()))
	}

	# create an example protocol buffer
	example = tf.train.Example(features = tf.train.Features(feature = feature))

	# Serialize to string and write on the file
	writer.write(example.SerializeToString())

writer.close()    # close the file using: writer.close()
sys.stdout.flush()

#===============================================#
#  Write validation data into a TFRecords file  #
#===============================================#
# open the TFRecord file
val_filename = 'mouth_val.tfrecords'      # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
	# Load the image
	img = load_image(os.path.join('./data',val_addrs[i]))
	label = val_labels[i]

	# create a feature
	feature = {
			'val/label':_int64_feature(label),
			'val/image':_bytes_feature(tf.compat.as_bytes(img.tostring()))
	}

	# create an example protocal buffer
	example = tf.train.Example(features = tf.train.Features(feature = feature))

	# Serialize to string and write on the file
	writer.write(example.SerializeToString())

writer.close()    
sys.stdout.flush()

#=========================================#
#  Write test data into a TFRecords file  #
#=========================================#
# open the TFRecord file
test_filename = 'mouth_test.tfrecords'      # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_addrs)):
	# Load the image
	img = load_image(os.path.join('./data',test_addrs[i]))
	label = test_labels[i]

	# create a feature
	feature = {
			'test/label':_int64_feature(label),
			'test/image':_bytes_feature(tf.compat.as_bytes(img.tostring()))
	}

	# create an example protocal buffer
	example = tf.train.Example(features = tf.train.Features(feature = feature))

	# Serialize to string and write on the file
	writer.write(example.SerializeToString())

writer.close()    
sys.stdout.flush()
# TFRecord

### 文件夹构成
```
   mouth____| |____README.md
			| |____tfrecord_reader.py
			| |____tfrecord_writer.py
			| |____data
				| |____223smile.jpg
				| |____86neutral.jpg
				   ...
```

### 我的运行环境

Python: 3.6.5

tf.__version__: '1.8.0'

### 运行方法
**1. 下载 mouth 数据**

1.1 数据集下载地址

链接：https://pan.baidu.com/s/1xPbIcPtYu4e-SYHJ4yu_8A 提取码：9x81 

1.2 数据集介绍

	- 这是一个嘴唇数据集，包括无状态的嘴唇和微笑状态的嘴唇。
	- 包括500张无状态的嘴唇和500张微笑的嘴唇，用于分类等任务.
	- 所有图片的尺寸为60x60。
	- 0 无状态	； 1 微笑

**2. Write data into a TFRecords file**

实验脚本：
```
python tfrecord_writer.py
```

**3. Read the TFRecords file**

实验脚本：
```
python tfrecord_reader.py
```

### 拓展阅读
本文参考：
1. https://blog.csdn.net/happyhorizion/article/details/77894055 特别详细的中文介绍 tensorflow读取数据-tfrecord格式
2. https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/ 
	- 英文整体介绍了什么是tfrecord，tfrecord的优点，还有生成tfrecord和解析tfrecord的大体过程
	- 生成TFRecord： Data -》 FeatureSet -》 Example -》 Serialized Example -》 tfrecord
	- 解析TFRecord： tfrecord -》 Serialized Example -》 Example -》 FeatureSet -》 Data
3. https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
	- 英文非常非常非常详细介绍了如何一步一步生成TFRecord和解析TFRecord，并且告诉了每一个函数的意思
	- 真的非常非常好，耐心看
4. http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
	- 英文How to write into and read from a TFRecords file in TensorFlow
	- 利用一个例子详细讲解，也非常非常好
5. 当然，github上搜索tfrecord也可以看到很多优秀的关于TFRecords的代码
	

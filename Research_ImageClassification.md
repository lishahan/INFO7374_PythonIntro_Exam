
# Exploratory Research on Image Classification using Tensorflow and Perceptual Hashing Algorithm


```python
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
import mysql.connector
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import csv
import datetime
import time
import re
import statsmodels.api as sm
import mysql
from prettytable import PrettyTable
import itertools
from itertools import chain
from PIL import Image
import imagehash
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
```

## Part1 Data Processing

### 1. Download dataset and store the label and image address information in a table of the data base created on Google Cloud Platform


```python
label1= pd.read_csv("/Users/lisha/Desktop/BigData/Assignment3/trainLabels.csv", sep=',')
```


```python
list_col=['ImageID','LabelName','Type','ImageAddr']
```


```python
df=pd.DataFrame(columns=list_col)
```


```python
df['ImageID']=label1['id']
df['LabelName']=label1['label']
df['Type']=label1['id']
```


```python
df.loc[df['ImageID']<=40000, 'Type'] = 'train'
df.loc[df['ImageID']>40000, 'Type'] = 'test'
```


```python
for i in range(len(df)):
    df.loc[i,'ImageAddr']='/Users/lisha/Desktop/train/'+ str(i+1) +'.png'
```


```python
df.to_csv("/Users/lisha/Desktop/ImageLabel.txt", sep=',',index=False, )
```


```python
# Open terminal to start proxy past this command: "./cloud_sql_proxy -instances=image-category-184805:us-east1:project7390=tcp:3306 &"
cnx = mysql.connector.connect(user='root', password='GoHusky!',
                              host='127.0.0.1',
                              database='ImageLabel')
c=cnx.cursor()
```


```python
c.execute("""DROP TABLE ImageLabel_local""")
```


```python
sql = """CREATE TABLE ImageLabel_local (ImageID int NOT NULL, LabelName  CHAR(20) NOT NULL, Type CHAR(10) NOT NULL, ImageAddr CHAR(255) NOT NULL)"""
c.execute(sql)
```


```python
query = "LOAD DATA LOCAL INFILE '/Users/lisha/Desktop/ImageLabel.txt' INTO TABLE ImageLabel_local FIELDS TERMINATED BY ','  Lines Terminated BY '\n' (ImageID,LabelName,Type,ImageAddr)"
c.execute( query )
cnx.commit()
```


```python
# A function to print returned data in a better form
def print_table(c):
    rows=c.fetchall()
    pt = PrettyTable([i[0] for i in c.description])
    pt.align= "l"
    for row in rows:
        pt.add_row(row)
    print (pt)
```


```python
# Database I created
c.execute("""SELECT * FROM ImageLabel_local LIMIT 10""")
print_table(c)
```

    +---------+------------+-------+-----------------------------------+
    | ImageID | LabelName  | Type  | ImageAddr                         |
    +---------+------------+-------+-----------------------------------+
    | 1       | frog       | train | /Users/lisha/Desktop/train/1.png  |
    | 2       | truck      | train | /Users/lisha/Desktop/train/2.png  |
    | 3       | truck      | train | /Users/lisha/Desktop/train/3.png  |
    | 4       | deer       | train | /Users/lisha/Desktop/train/4.png  |
    | 5       | automobile | train | /Users/lisha/Desktop/train/5.png  |
    | 6       | automobile | train | /Users/lisha/Desktop/train/6.png  |
    | 7       | bird       | train | /Users/lisha/Desktop/train/7.png  |
    | 8       | horse      | train | /Users/lisha/Desktop/train/8.png  |
    | 9       | ship       | train | /Users/lisha/Desktop/train/9.png  |
    | 10      | cat        | train | /Users/lisha/Desktop/train/10.png |
    +---------+------------+-------+-----------------------------------+


### 2. Use Dataset API to process data suitable for tensorflow


```python
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 28, 28)
    return image_resized, label
```


```python
def get_dataset(list_file, list_label):
    filenames = tf.constant(list_file)
    labels = tf.constant(list_label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    return dataset
```


```python
def tuple_list(tuplelist):
    return list(itertools.chain(*tuplelist))
```


```python
c=cnx.cursor()
c.execute("""SELECT LabelName FROM ImageLabel_local WHERE LabelName='cat'""")
list_label_cat=tuple_list(c.fetchall())
c.execute("""SELECT LabelName FROM ImageLabel_local WHERE LabelName='dog'""")
list_label_dog=tuple_list(c.fetchall())
c.execute("""SELECT LabelName FROM ImageLabel_local WHERE LabelName='truck'""")
list_label_truck=tuple_list(c.fetchall())
c.execute("""SELECT LabelName FROM ImageLabel_local""")
list_label=tuple_list(c.fetchall())
```


```python
c=cnx.cursor()
c.execute("""SELECT ImageAddr FROM ImageLabel_local WHERE LabelName='cat'""")
list_file_cat=tuple_list(c.fetchall())
c.execute("""SELECT ImageAddr FROM ImageLabel_local WHERE LabelName='dog'""")
list_file_dog=tuple_list(c.fetchall())
c.execute("""SELECT ImageAddr FROM ImageLabel_local WHERE LabelName='truck'""")
list_file_truck=tuple_list(c.fetchall())
c.execute("""SELECT ImageAddr FROM ImageLabel_local""")
list_file=tuple_list(c.fetchall())
```


```python
dataset_all=get_dataset(list_file, list_label)
dataset_cat=get_dataset(list_file_cat, list_label_cat)
dataset_dog=get_dataset(list_file_dog, list_label_dog)
dataset_truck=get_dataset(list_file_truck, list_label_truck)
```


```python
dataset_all
```




    <MapDataset shapes: ((28, 28, ?), ()), types: (tf.uint8, tf.string)>




```python
batched_dataset = dataset_all.batch(100)
```

## Part2 Exploring possibility of using perceputal hashing algorithm in image classification

### 1. Take samples from dataset and see hamming distance between same/different classes


```python
# functions used to get hashing value and hamming distance
def get_hash(image):
    return str(imagehash.average_hash(Image.open(image)))

def get_hashlist(images):
    list_hash=[]
    for image in images:
        list_hash.append(get_hash(image))
    return list_hash

def hamming(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

def hamming_list(hash_list1, hash_list2):
    dif_list=[]
    i=0
    j=0
    for i in range(len(hash_list1)):
        for j in range(len(hash_list2)):
            dif_list.append(hamming(hash_list1[i],hash_list2[j]))
    return dif_list
```


```python
list_cat=get_hashlist(list_file_cat)
list_dog=get_hashlist(list_file_dog)
list_truck=get_hashlist(list_file_truck)
```


```python
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import warnings
```


```python
plotly.tools.set_credentials_file(username='asahoho', api_key='9ibe6uhTBjIiY4eNOym2')
```


```python
N=100
trace0 = go.Scatter(
    x = np.random.randn(N),
    y = hamming_list(list_cat[:100], list_cat[:100]),
    name = 'cat-cat',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 1,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = np.random.randn(N),
    y = hamming_list(list_dog[:100], list_dog[:100]),
    name = 'dog-dog',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 1
        )
    )
)

trace2 = go.Scatter(
    x = np.random.randn(N),
    y = hamming_list(list_dog[:100], list_truck[:100]),
    name = 'truck-dog',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'white',
        line = dict(
            width = 1,
            color = 'rgb(0, 0, 0)'
        )
    )
)

data = [trace0, trace1, trace2]
layout = dict(
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')
```

    High five! You successfully sent some data to your account on plotly. View your plot in your browser at https://plot.ly/~asahoho/0 or inside your plot.ly account where it is named 'styled-scatter'





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~asahoho/0.embed" height="525px" width="100%"></iframe>



### 2. Buidl a CNN model using tensorflow
#### See if lower quality images dataset will get lower accuracy


```python
def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image
```


```python
def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
# Layer 1
    with tf.variable_scope('Layer1', reuse = tf.AUTO_REUSE):
# Define the variables
        weights = tf.get_variable(
        name='weights',
        shape=[image_pixels, hidden_units],
        initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(image_pixels))),
        regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
    
        biases = tf.Variable(tf.zeros([hidden_units]), name='biases')
# Define the layer's output
        hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

# Layer 2
    with tf.variable_scope('Layer2', reuse = tf.AUTO_REUSE):
# Define variables
        weights = tf.get_variable('weights', [hidden_units, classes],
        initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden_units))),
        regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
    
        biases = tf.Variable(tf.zeros([classes]), name='biases')
# Define the layer's output
        logits = tf.matmul(hidden, weights) + biases
# Define summery-operation for 'logits'-variable
        tf.summary.histogram('logits', logits)
    
    return logits
```


```python
def loss(logits, labels):
    with tf.name_scope('Loss'):
# Operation to determine the cross entropy between logits and labels
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
# Operation for the loss function
        loss = cross_entropy + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
# Add a scalar summary for the loss
        tf.summary.scalar('loss', loss)
    
    return loss
```


```python
def training(loss, learning_rate):
# Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
# Create a gradient descent optimizer (which also increments the global step counter)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    return train_step
```


```python
def evaluation(logits, labels):
    with tf.name_scope('Accuracy'):
# Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)
# Operation calculating the accuracy of the predictions
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Summary operation for the accuracy
        tf.summary.scalar('train_accuracy', accuracy)
    
    return accuracy
```


```python
def gen_batch(data, batch_size, num_iter):
    data = np.array(data)
    index = len((data))
    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data)):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
    yield data[index:index + batch_size]
    #return data
```


```python
import sys
import pickle

def load_CIFAR10_batch(filename):
    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            dict = pickle.load(f)
        else:
            dict = pickle.load(f, encoding='latin1')
        x = dict['data']
        y = dict['labels']
        x = x.astype(float)
        y = np.array(y)
    return x, y

def load_data():
    xs = []
    ys = []
    for i in range(1, 6):
        filename = 'cifar-10-batches-py/data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys
    x_test, y_test = load_CIFAR10_batch('cifar-10-batches-py/test_batch')
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']
# Normalize Data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'images_test': x_test,
    'labels_test': y_test,
    'classes': classes
  }
    return reshape_data(data_dict)

def reshape_data(data_dict):
    im_tr = np.array(data_dict['images_train'])
    im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
    im_tr = np.transpose(im_tr, (0,2,3,1))
    data_dict['images_train'] = im_tr
    im_te = np.array(data_dict['images_test'])
    im_te = np.reshape(im_te, (-1, 3, 32, 32))
    im_te = np.transpose(im_te, (0,2,3,1))
    data_dict['images_test'] = im_te
    return data_dict
```


```python
data_sets = load_data()
```


```python
# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr, value))
print()
```

    
    Parameters:
    batch_size = 400
    hidden1 = 120
    learning_rate = 0.001
    max_steps = 2000
    reg_constant = 0.1
    train_dir = tf_logs
    



```python
IMAGE_PIXELS = 3072
CLASSES = 10
```


```python
# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None, 10], name='image-labels')
```


```python
# Operation for the classifier's result
logits = inference(images_placeholder, IMAGE_PIXELS, FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)
```


```python
# Operation for the loss function
loss = loss(logits, labels_placeholder)
# Operation for the training step
train_step = training(loss, FLAGS.learning_rate)
# Operation calculating the accuracy of our predictions
accuracy = evaluation(logits, labels_placeholder)
```


```python
summary = tf.summary.merge_all()
saver = tf.train.Saver()
```


```python
sess=tf.Session()
# Initialize variables and create summary-writer
sess.run(tf.initialize_all_variables())
summary_writer = tf.summary.FileWriter('/Users/lisha/Desktop', sess.graph)
```


```python
# Generate input data batches
zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
batches = gen_batch(zipped_data, FLAGS.batch_size, FLAGS.max_steps)
```


```python
try:
    for i in range(FLAGS.max_steps):
# Get next input data batch
        batch = batches.__next__()
        images_batch, labels_batch = zip(*batch)
        feed_dict = {
          images_placeholder: images_batch,
          labels_placeholder: labels_batch
        }
except StopIteration:
    pass
```


```python
feed_dict = {
          images_placeholder: data_sets['images_train'][:FLAGS.batch_size],
          labels_placeholder: data_sets['labels_train'][:FLAGS.batch_size]
        }
```


```python
sess.run(tf.global_variables_initializer())
```

### The following part is not done yet. Too much errors to be fixed.
### There is an important part is to recognize low quality images dataset to see if the accuracy of this model dropped.


```python
# Periodically print out the model's current accuracy
if i % 100 == 0:
    train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
    summary_str = sess.run(summary, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, i)
```


```python
# Perform a single training step
sess.run([train_step, loss], feed_dict=feed_dict)
```


```python
# Periodically save checkpoint
if (i + 1) % 1000 == 0:
    checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=i)
print('Saved checkpoint')
```


```python
# After finishing the training, evaluate on the test set
test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test']})
print('Test accuracy {:g}'.format(test_accuracy))
```

# What is next?

## Part3 Use perceptual hashing to generate dataset to feed in the model(small batch)
## Part4 Run the whole dataset on Google Cloud Platform to see if there is any improvement on the model

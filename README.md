# Distributed

## Python

Prerequisites: 4 machines with direct network connection, make sure `ping <ip>` works across them.

We will use [TensorFlow](https://www.tensorflow.org) and Keras since it's currently the only supported strategy for
[MultiWorkerMirroredStrategy](https://www.tensorflow.org/guide/distributed_training#types_of_strategies).

First install depedencies and validate versions,

```bash
pip install --upgrade pip
pip install -q tf-nightly --user
pip install -q tfds-nightly --user
```
```python
import tensorflow as tf; print(tf.__version__)
```
```
2.2.0-dev20200410
```
```python
import tensorflow_datasets as tfds; print(tfds.__version__)
```
```
2.1.0
```

### Local

To validate the configuration in each worker node is correct, we will first train locally over each worker. First import TensorFlow and [TensorFlow Datasets](https://www.tensorflow.org/datasets),

```python
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()
```

Which we can now use to download [MNIST](http://yann.lecun.com/exdb/mnist/),

```python
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

train_datasets = make_datasets_unbatched().batch(BATCH_SIZE)
```

We can now define a network,

```python
def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model
```

And train over a single worker to validate training is working properly,

```python
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)
```

### Distributed

Now moving on to distributed training, we first need to define a configuration file for each worker node as follows, you should restart your Python session after running the local validation steps. Notice that different configurations are applied over each worker node.

```python
# run in main worker
import os
import json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089"]
    },
    'task': {'type': 'worker', 'index': 0}
})

# run in worker(1)
import os
import json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089"]
    },
    'task': {'type': 'worker', 'index': 1}
})

# run in worker(2)
import os
import json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089"]
    },
    'task': {'type': 'worker', 'index': 2}
})

# run in worker(3)
import os
import json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089"]
    },
    'task': {'type': 'worker', 'index': 3}
})
```

We can then define the `MultiWorkerMirroredStrategy` strategy across all workers,

```python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```
```
INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:0/device:CPU:0', '/job:worker/replica:0/task:0/device:XLA_CPU:0']
INFO:tensorflow:Using MirroredStrategy with devices ('/job:worker/task:0',)
INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
```

And the data processing and model definition,

```python
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model
```

That's it, we should now run the following code over each worker node to trigger training distributed,

```python
NUM_WORKERS = 4
BUFFER_SIZE = 10000

# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

# Creation of dataset needs to be after MultiWorkerMirroredStrategy object
# is instantiated.
train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)
```
```
INFO:tensorflow:Running Distribute Coordinator with mode = 'independent_worker', cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, environment = None, rpc_layer = 'grpc'
INFO:tensorflow:Running Distribute Coordinator with mode = 'independent_worker', cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, environment = None, rpc_layer = 'grpc'
WARNING:tensorflow:`eval_fn` is not passed in. The `worker_fn` will be used if an "evaluator" task exists in the cluster.
WARNING:tensorflow:`eval_fn` is not passed in. The `worker_fn` will be used if an "evaluator" task exists in the cluster.
WARNING:tensorflow:`eval_strategy` is not passed in. No distribution strategy will be used for evaluation.
WARNING:tensorflow:`eval_strategy` is not passed in. No distribution strategy will be used for evaluation.
INFO:tensorflow:Using MirroredStrategy with devices ('/job:worker/task:0',)
INFO:tensorflow:Using MirroredStrategy with devices ('/job:worker/task:0',)
INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
INFO:tensorflow:Using MirroredStrategy with devices ('/job:worker/task:0',)
INFO:tensorflow:Using MirroredStrategy with devices ('/job:worker/task:0',)
INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['172.17.0.6:10090', '172.17.0.3:10088', '172.17.0.4:10087', '172.17.0.5:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
Epoch 1/3
INFO:tensorflow:Collective batch_all_reduce: 6 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 6 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 6 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 6 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4, communication_hint = AUTO, num_packs = 1
5/5 [==============================] - 1s 130ms/step - accuracy: 0.0930 - loss: 2.3127
Epoch 2/3
5/5 [==============================] - 0s 84ms/step - accuracy: 0.1070 - loss: 2.3080
Epoch 3/3
5/5 [==============================] - 0s 76ms/step - accuracy: 0.1102 - loss: 2.3040
```

Adapted from [Multi-worker training with Keras
](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).

## R

When using R, we will also make sure the workers are prooperly configured by training a local model first and installing all the required packages:

```r
install.packages("tenesorflow")
install.packages("keras")
install.packages("remotes")
remotes::install_github("rstudio/tfds")
```

We can then define our model,

```r
library(keras)
library(tfds)

BUFFER_SIZE <- 10000
BATCH_SIZE <- 64

mnist <- tfds_load("mnist")

train_dataset <- mnist$train %>% 
  dataset_map(function(record) {
    record$image <- tf$cast(record$image, tf$float32) / 255
    record}) %>%
  dataset_cache() %>%
  dataset_shuffle(BUFFER_SIZE) %>% 
  dataset_batch(BATCH_SIZE) %>% 
  dataset_map(unname)
  
model <- keras_model_sequential() %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = 3,
    activation = 'relu',
    input_shape = c(28, 28, 1)
  ) %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
   
model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy')
```

Then go ahead and train this local model,

```r
model %>% fit(train_dataset, epochs = 3)
```

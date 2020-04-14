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

### Local

When using R, we will also make sure the workers are prooperly configured by training a local model first and installing all the required R packages:

```r
install.packages("tenesorflow")
install.packages("keras")
install.packages("remotes")
remotes::install_github("rstudio/tfds", ref = "bugfix/in-memory-api")
remotes::install_github("rstudio/keras", ref = "bugfix/chief-worker")
```

And the required runtime dependencies:

```r
tensorflow::install_tensorflow(version = "nightly")
tfds::install_tfds()
tensorflow::tf_version()
```
```
Using virtual environment '~/.virtualenvs/r-reticulate' ...
[1] ‘2.2’
```

We can then define our model,

```r
library(tensorflow)
library(keras)

library(tfdatasets)
library(tfds)

BUFFER_SIZE <- 10000L
BATCH_SIZE <- 64L

mnist <- tfds_load("mnist")

train_dataset <- mnist$train %>% 
  dataset_map(function(record) {
    record$image <- tf$cast(record$image, tf$float32) / 255
    record}) %>%
  dataset_cache() %>%
  dataset_shuffle(BUFFER_SIZE) %>% 
  dataset_batch(BATCH_SIZE) %>% 
  dataset_map(unname)

result <- tfds:::tfds$load(name = "mnist", with_info = TRUE, as_supervised = TRUE)
datasets <- result[[1]]
info <- result[[2]]

train_dataset <- datasets$train$map(function(image, label) {
        image <- tf$cast(image, tf$float32)
        image <- image / 255
        list(image, label)
    })$cache()$shuffle(BUFFER_SIZE)$batch(BATCH_SIZE)
  
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

### Distributed

Let's go now for distributed training, but first restart your R session, then define the cluster specification in each worker:

```r
# run in main worker
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089")
    ),
    task = list(type = 'worker', index = 0)
), auto_unbox = TRUE))

# run in worker(1)
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089")
    ),
    task = list(type = 'worker', index = 1)
), auto_unbox = TRUE))

# run in worker(2)
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089")
    ),
    task = list(type = 'worker', index = 2)
), auto_unbox = TRUE))

# run in worker(3)
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.17.0.6:10090", "172.17.0.3:10088", "172.17.0.4:10087", "172.17.0.5:10089")
    ),
    task = list(type = 'worker', index = 3)
), auto_unbox = TRUE))
```

We can now redefine out models using a MultiWorkerMirroredStrategy strategy as follows:

```r
library(tensorflow)
library(keras)

library(tfdatasets)
library(tfds)

BUFFER_SIZE <- 10000L
NUM_WORKERS <- 4L
BATCH_SIZE <- 64L * NUM_WORKERS

strategy <- tf$distribute$experimental$MultiWorkerMirroredStrategy()

result <- tfds:::tfds$load(name = "mnist", with_info = TRUE, as_supervised = TRUE)
datasets <- result[[1]]
info <- result[[2]]

train_dataset <- datasets$train$map(function(image, label) {
        image <- tf$cast(image, tf$float32)
        image <- image / 255
        list(image, label)
    })$cache()$shuffle(BUFFER_SIZE)$batch(BATCH_SIZE)

with (strategy$scope(), {
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
})
```

Finally, we can train across all workers by running over each of them,

```r
model %>% fit(train_dataset, epochs = 3)
```

**Note:** Currently error when consuming dataset

```
Error in py_call_impl(callable, dots$args, dots$keywords): InvalidArgumentError:  There aren't enough elements in this dataset for each shard to have at least one element (# elems = 1, # shards = 4). If you are using datasets with distribution strategy, considering setting the auto sharding policy to either DATA or OFF using the `experimental_distribute.auto_shard_policy` optionof `tf.data.Options()`.
	 [[{{node MultiDeviceIteratorGetNextFromShard}}]]
	 [[RemoteCall]]
	 [[IteratorGetNext]] [Op:__inference_distributed_function_1135]

Function call stack:
distributed_function


Detailed traceback: 
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training.py", line 819, in fit
    use_multiprocessing=use_multiprocessing)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_distributed.py", line 790, in fit
    *args, **kwargs)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_distributed.py", line 777, in wrapper
    mode=dc.CoordinatorMode.INDEPENDENT_WORKER)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/distribute/distribute_coordinator.py", line 853, in run_distribute_coordinator
    task_id, session_config, rpc_layer)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/distribute/distribute_coordinator.py", line 360, in _run_single_worker
    return worker_fn(strategy)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_distributed.py", line 772, in _worker_fn
    return method(model, **kwargs)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_v2.py", line 342, in fit
    total_epochs=epochs)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_v2.py", line 128, in run_one_epoch
    batch_outs = execution_function(iterator)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py", line 98, in execution_function
    distributed_function(input_fn))
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/def_function.py", line 568, in __call__
    result = self._call(*args, **kwds)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/def_function.py", line 632, in _call
    return self._stateless_fn(*args, **kwds)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 2363, in __call__
    return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1611, in _filtered_call
    self.captured_inputs)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1692, in _call_flat
    ctx, args, cancellation_manager=cancellation_manager))
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 545, in call
    ctx=ctx)
  File "/home/rstudio/.virtualenvs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from

Traceback:

1. model %>% fit(train_dataset, epochs = 3)
2. withVisible(eval(quote(`_fseq`(`_lhs`)), env, env))
3. eval(quote(`_fseq`(`_lhs`)), env, env)
4. eval(quote(`_fseq`(`_lhs`)), env, env)
5. `_fseq`(`_lhs`)
6. freduce(value, `_function_list`)
7. withVisible(function_list[[k]](value))
8. function_list[[k]](value)
9. fit(., train_dataset, epochs = 3)
10. fit.keras.engine.training.Model(., train_dataset, epochs = 3)
11. do.call(object$fit, args)
12. (structure(function (...) 
  . {
  .     dots <- py_resolve_dots(list(...))
  .     result <- py_call_impl(callable, dots$args, dots$keywords)
  .     if (convert) 
  .         result <- py_to_r(result)
  .     if (is.null(result)) 
  .         invisible(result)
  .     else result
  . }, class = c("python.builtin.method", "python.builtin.object"
  . ), py_object = <environment>))(batch_size = NULL, epochs = 3L, 
  .     verbose = 1L, callbacks = list(<environment>), validation_split = 0, 
  .     shuffle = TRUE, class_weight = NULL, sample_weight = NULL, 
  .     initial_epoch = 0L, x = <environment>)
13. py_call_impl(callable, dots$args, dots$keywords)
```

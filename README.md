# Distributed

For now, just a README with resources on running TensorFlow with distributed resources using R, but Python code also provided for reference.

## Prerequisites

THis README assumes you have access to four different machines connected to the same network. You can easily get 4 EC2 AMI machines and install R/RStudio as follows:

```
sudo yum update
sudo amazon-linux-extras install R3.4
sudo yum install python3

wget https://s3.amazonaws.com/rstudio-ide-build/server/centos6/x86_64/rstudio-server-rhel-1.3.947-x86_64.rpm
sudo yum install rstudio-server-rhel-1.3.947-x86_64.rpm

sudo useradd rstudio
sudo passwd rstudio
```

## R

### Local

When using R, we will also make sure the workers are prooperly configured by training a local model first and installing all the required R packages:

```r
install.packages("tensorflow")
install.packages("remotes")
remotes::install_github("rstudio/keras")
```

And the required runtime dependencies:

```r
tensorflow::install_tensorflow()
tensorflow::tf_version()
```
```
Using virtual environment '~/.virtualenvs/r-reticulate' ...
[1] ‘2.0’
```

We can then define and train our model,

```r        
library(tensorflow)
library(keras)

batch_size <- 64L

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_train <- x_train / 255

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
    layer_dense(units = 10)

model %>% compile(
  loss = tf$keras$losses$SparseCategoricalCrossentropy(from_logits = TRUE),
  optimizer = tf$keras$optimizers$SGD(learning_rate = 0.001),
  metrics = 'accuracy')

model %>% fit(x_train, y_train, batch_size = batch_size, epochs = 3, steps_per_epoch = 5)
```

### Distributed

Let's go now for distributed training, but first restart your R session, then define the cluster specification in each worker:

```r
# run in main worker
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.31.11.122:10090", "172.31.10.143:10088", "172.31.4.119:10087", "172.31.4.116:10089")
    ),
    task = list(type = 'worker', index = 0)
), auto_unbox = TRUE))

# run in worker(1)
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.31.11.122:10090", "172.31.10.143:10088", "172.31.4.119:10087", "172.31.4.116:10089")
    ),
    task = list(type = 'worker', index = 1)
), auto_unbox = TRUE))

# run in worker(2)
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.31.11.122:10090", "172.31.10.143:10088", "172.31.4.119:10087", "172.31.4.116:10089")
    ),
    task = list(type = 'worker', index = 2)
), auto_unbox = TRUE))

# run in worker(3)
Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
    cluster = list(
        worker = c("172.31.11.122:10090", "172.31.10.143:10088", "172.31.4.119:10087", "172.31.4.116:10089")
    ),
    task = list(type = 'worker', index = 3)
), auto_unbox = TRUE))
```

We can now redefine out models using a MultiWorkerMirroredStrategy strategy as follows:

```r
library(tensorflow)
library(keras)

strategy <- tf$distribute$experimental$MultiWorkerMirroredStrategy()

num_workers <- 4L
batch_size <- 64L * num_workers

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_train <- x_train / 255

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
      layer_dense(units = 10)

  model %>% compile(
    loss = tf$keras$losses$SparseCategoricalCrossentropy(from_logits = TRUE),
    optimizer = tf$keras$optimizers$SGD(learning_rate = 0.001),
    metrics = 'accuracy')
})

model %>% fit(x_train, y_train, batch_size = batch_size, epochs = 3, steps_per_epoch = 5)
```

### Spark

When using Apache Spark, first install the dependencies as follows in the driver node:

```r
install.packages("tensorflow")
install.packages("remotes")

remotes::install_github("rstudio/keras")
remotes::install_github("sparklyr/sparklyr")
```

Then you can connect to Spark as usual using `spark_connet()`, followed by using `spark_apply()` with `barrier = TRUE`:

```r
library(sparklyr)
sc <- spark_connect(master = "yarn", spark_home = "/usr/lib/spark/", config = list(spark.dynamicAllocation.enabled = FALSE, `sparklyr.shell.executor-cores` = 8, `sparklyr.shell.num-executors` = 3, sparklyr.apply.env.WORKON_HOME = "/tmp/.virtualenvs"))

sdf_len(sc, 3, repartition = 3) %>%
  spark_apply(function(df, barrier) {
    tryCatch({
      library(tensorflow)
      library(keras)
      
      Sys.setenv(TF_CONFIG = jsonlite::toJSON(list(
        cluster = list(worker = paste(gsub(":[0-9]+$", "", barrier$address), 8000 + seq_along(barrier$address), sep = ":")),
        task = list(type = 'worker', index = barrier$partition)
      ), auto_unbox = TRUE))
      
      if (is.null(tf_version())) install_tensorflow()
      
      strategy <- tf$distribute$experimental$MultiWorkerMirroredStrategy()
      
      num_workers <- 3L
      batch_size <- 64L * num_workers
      
      mnist <- dataset_mnist()
      x_train <- mnist$train$x
      y_train <- mnist$train$y
      
      x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
      x_train <- x_train / 255
      
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
          layer_dense(units = 10)
        
        model %>% compile(
          loss = tf$keras$losses$SparseCategoricalCrossentropy(from_logits = TRUE),
          optimizer = tf$keras$optimizers$SGD(learning_rate = 0.001),
          metrics = 'accuracy')
      })
      
      result <- model %>% fit(x_train, y_train, batch_size = batch_size, epochs = 3, steps_per_epoch = 5)
      
      as.character(max(result$metrics$accuracy))
    }, error = function(e) e$message)
  }, barrier = TRUE, columns = c(address = "character"), ) %>%
  collect()
```
```
# A tibble: 3 x 1
  address         
  <chr>           
1 0.11979166418314
2 0.11979166418314
3 0.11979166418314
```

To retrieve the model, instead of returning accuracy or other metrics you can serialize the HDF5 file and retrieve it form one of the worker nodes as follows,

```r
model_file <- paste0("trained-", barrier$partition, ".hdf5")
save_model_hdf5(model, model_file)

if (barrier$partition == 0) base64enc::base64encode(model_file) else ""
```

Which you can then save in the driver node and use it later on to perform scoring, etc.

```r
write(base64enc::base64decode(result$address[1]), "model.hdf5")
```

## Python

Prerequisites: 4 machines with direct network connection, make sure `ping <ip>` works across them.

We will use [TensorFlow](https://www.tensorflow.org) and Keras since it's currently the only supported strategy for
[MultiWorkerMirroredStrategy](https://www.tensorflow.org/guide/distributed_training#types_of_strategies).

First install depedencies and validate versions,

```bash
pip install --upgrade pip
pip install -q tensorflow==2.0 --user
```
```python
import tensorflow as tf; print(tf.__version__)
```
```
2.0.0
```

You might also want to find the IP in each worker node as follows,

```python
import socket    
hostname = socket.gethostname()    
socket.gethostbyname(hostname)
```

### Local

To validate the configuration in each worker node is correct, we will first train locally over each worker [MNIST](http://yann.lecun.com/exdb/mnist/),

```python
import tensorflow as tf

batch_size = 64

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = 3, steps_per_epoch = 5)
```
```
Train on 60000 samples
Epoch 1/3
  320/60000 [..............................] - ETA: 2:25 - loss: 2.2995 - accuracy: 0.2062 Epoch 2/3
  192/60000 [..............................] - ETA: 29s - loss: 2.3059 - accuracy: 0.2240Epoch 3/3
  128/60000 [..............................] - ETA: 37s - loss: 2.2968 - accuracy: 0.1953
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
import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

num_workers = 4
batch_size = 64 * num_workers

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

with strategy.scope():
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
```

And then train the model

```python
model.fit(x_train, y_train, batch_size = batch_size, epochs = 3, steps_per_epoch = 5)
```
```
INFO:tensorflow:Running Distribute Coordinator with mode = 'independent_worker', cluster_spec = {'worker': ['172.17.0.3:10090', '172.17.0.4:10088', '172.17.0.5:10087', '172.17.0.6:10089']}, task_type = 'worker', task_id = 0, environment = None, rpc_layer = 'grpc'
WARNING:tensorflow:`eval_fn` is not passed in. The `worker_fn` will be used if an "evaluator" task exists in the cluster.
WARNING:tensorflow:`eval_strategy` is not passed in. No distribution strategy will be used for evaluation.
INFO:tensorflow:Multi-worker CollectiveAllReduceStrategy with cluster_spec = {'worker': ['172.17.0.3:10090', '172.17.0.4:10088', '172.17.0.5:10087', '172.17.0.6:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
INFO:tensorflow:Multi-worker CollectiveAllReduceStrategy with cluster_spec = {'worker': ['172.17.0.3:10090', '172.17.0.4:10088', '172.17.0.5:10087', '172.17.0.6:10089']}, task_type = 'worker', task_id = 0, num_workers = 4, local_devices = ('/job:worker/task:0',), communication = CollectiveCommunication.AUTO
WARNING:tensorflow:ModelCheckpoint callback is not provided. Workers will need to restart training if any fails.
Train on 60000 samples
Epoch 1/3
INFO:tensorflow:Collective batch_all_reduce: 6 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 6 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
INFO:tensorflow:Collective batch_all_reduce: 1 all-reduces, num_workers = 4
 1280/60000 [..............................] - ETA: 7:27 - loss: 2.3116 - accuracy: 0.0766 Epoch 2/3
 1280/60000 [..............................] - ETA: 9s - loss: 2.3067 - accuracy: 0.0766 Epoch 3/3
 1280/60000 [..............................] - ETA: 9s - loss: 2.3091 - accuracy: 0.0719
```


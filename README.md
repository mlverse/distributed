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

We should now import TensorFlow and [TensorFlow Datasets](https://www.tensorflow.org/datasets),

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

Adapted from [Multi-worker training with Keras
](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).

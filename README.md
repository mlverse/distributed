# Distributed

## Python

Prerequisites: 4 machines with direct network connection, make sure `ping <ip>` works across them.

We will use TensorFlow and Keras since it's currently the only supported strategy for
[MultiWorkerMirroredStrategy](https://www.tensorflow.org/guide/distributed_training#types_of_strategies).

```python
pip install --upgrade pip
pip install -q tf-nightly --user
import tensorflow as tf; print(tf.__version__)
```
```
2.2.0-dev20200410
```

Adapted from [Multi-worker training with Keras
](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).

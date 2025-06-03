import pytest
# ------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("size", [5000, 10000])
def test_gpu_malMut(size):
    import time
    import tensorflow as tf
    from .utils import checkGpu
    
    checkGpu()

    with tf.device('CPU:0'):
        start = time.time()
        # enter code here of tf data
        v1 = tf.Variable(tf.random.normal((size, size)))
        v2 = tf.Variable(tf.random.normal((size, size)))
        op = tf.matmul(v1, v2)

        tcpu = time.time() - start

        print(time.time() - start)

    with tf.device('GPU:0'):
        start = time.time()
        # enter code here of tf data
        v1 = tf.Variable(tf.random.normal((size, size)))
        v2 = tf.Variable(tf.random.normal((size, size)))
        op = tf.matmul(v1, v2)

        tgpu = time.time() - start

        print(time.time() - start)

    if tgpu > tcpu:
        msg = ("Time on gpu: {:0.4f} s ".format(tgpu) +
               "Time on: {:0.4f} s".format(tcpu))
        raise Exception(msg)

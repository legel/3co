import tensorflow as tf

#assert tf.test.is_gpu_available()
#assert tf.test.is_built_with_cuda()
tf.test.gpu_device_name()
#tf.config.list_physical_devices('GPU')
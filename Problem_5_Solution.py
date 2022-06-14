from code_for_hw8_keras import *
from keras.initializers.initializers_v2 import VarianceScaling
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train, validation = get_MNIST_data()
# Problem 5B
layers = [Dense(input_dim=28*28, units=10, activation='softmax')]
# Problem 5C
# layers_1 = [Dense(input_dim=28*28, units=10, activation='softmax',
#                   kernel_initializer=VarianceScaling(scale=0.001, mode='fan_in', distribution='normal',
#                                                      seed=None),)]
run_keras_fc_mnist(train, validation, layers, 1, split=0.1, verbose=False, trials=5)


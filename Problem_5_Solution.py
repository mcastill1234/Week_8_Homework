from code_for_hw8_keras import *
import tensorflow as tf
from keras.initializers.initializers_v2 import VarianceScaling
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

train, validation = get_MNIST_data()
train_n = train[0] / 255, train[1]
validation = validation[0] / 255, validation[1]

# Problem 5B: Uncomment run_keras_fc function below to run this problem
layers = [Dense(input_dim=28*28, units=10, activation='softmax')]
# run_keras_fc_mnist(train, validation, layers, 5, split=0.1, verbose=False, trials=1)

# Problem 5C: Uncomment run_keras_fc function below to run this problem
layers_1 = [Dense(input_dim=28*28, units=10, activation='softmax',
                  kernel_initializer=VarianceScaling(scale=0.001, mode='fan_in', distribution='normal',
                                                     seed=None),)]
# run_keras_fc_mnist(train, validation, layers_1, 5, split=0.1, verbose=False, trials=1)

# Problem 5H: Change the units parameter from 128 to 1024 to get the problem solutions.
layers_2 = [Dense(input_dim=28*28, units=1024, activation='relu'),
            Dense(units=10, activation='softmax')]

# run_keras_fc_mnist(train_n, validation, layers_2, 1, split=0.1, verbose=False, trials=1)

# Problem 5I: Problem 5J: Uncomment run_keras_fc function below to run this problem
layers_3 = [Dense(input_dim=28*28, units=512, activation='relu'),
            Dense(units=256, activation='relu'),
            Dense(units=10, activation='softmax')]

# run_keras_fc_mnist(train_n, validation, layers_3, 1, split=0.1, verbose=False, trials=1)

# Problem 5J: Uncomment run_keras_cnn function below to run this problem
layers_4 = [Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(0.5),
            Dense(units=10, activation='softmax')]

# run_keras_cnn_mnist(train_n, validation, layers_4, 1, split=0.1, verbose=False, trials=1)

# Problem 5K
train_20, validation_20 = get_MNIST_data(shift=20)
train_20 = train_20[0] / 255, train_20[1]
validation_20 = validation_20[0] / 255, validation_20[1]

layers_5 = [Dense(input_dim=48*48, units=512, activation='relu'),
            Dense(units=256, activation='relu'),
            Dense(units=10, activation='softmax')]

layers_6 = [Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(0.5),
            Dense(units=10, activation='softmax')]

run_keras_fc_mnist(train_20, validation_20, layers_5, 1, split=0.1, verbose=False, trials=1)
run_keras_cnn_mnist(train_20, validation_20, layers_6, 1, split=0.1, verbose=False, trials=1)


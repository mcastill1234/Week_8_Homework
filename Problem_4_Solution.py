from code_for_hw8_keras import *
from keras.layers import Conv1D, MaxPooling1D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Problem 4B: Code template if you would like to check 4B) through code
imsize = 1024
prob_white = 0.1
batch = 1
tsize = 1000
num_filters = 1  # Your code
kernel_size = 2  # Your code
strides = 1  # Your code
activation_conv = 'relu'  # Your code

(X_train, Y_train, X_val, Y_val, X_test, Y_test) = get_image_data_1d(1000, imsize, prob_white)

layer1 = Conv1D(filters=num_filters, kernel_size=kernel_size, strides=strides, use_bias=False,
                activation=activation_conv, batch_size=1, input_shape=(imsize, 1), padding='same')

activation_dense = 'linear'  # Your code
num_units = 1  # Your code
layer3 = Dense(units=num_units, activation=activation_dense, use_bias=False)

layers = [layer1, Flatten(), layer3]

# This is how we create the model using our layers
# model = Sequential()
# for layer in layers:
#     model.add(layer)
#
# model.compile(loss='mse', optimizer=Adam())

# Set the weights of the layers to desired values. We give you the lines to use for this part
# model.layers[0].set_weights([np.array([1/2, 1/2]).reshape(2, 1, 1)])
# model.layers[-1].set_weights([(np.ones(imsize)*10).reshape(imsize, 1)])
#
# model.evaluate(X_test, Y_test)

# Problem 4C: Set use_bias in the last layer to True and replace above with the following line:
# model.layers[-1].set_weights([np.ones(imsize).reshape(imsize,1),np.array([-10])])

# Problem 4E:
# trials = 5
# for trial in range(trials):
#     model, err = train_neural_counter(layers, data, 'mse')
#     print(err)

# Problem 4F
# data = get_image_data_1d(tsize, imsize, 0.1)
# trials = 5
# for trial in range(trials):
#     layer1 = Conv1D(filters=1, kernel_size=kernel_size, kernel_regularizer=filter_reg, strides=strides,
#                     use_bias=False, activation='relu', batch_size=batch, input_shape=(imsize, 1), padding='same')
#     layer3 = Dense(units=1, activation='linear', use_bias=False)
#     layers_1 = [layer1, Flatten(), layer3]
#     model, err = train_neural_counter(layers_1, data, 'mse')
#     print(err)

# Problem 4I
imsize = 128
data = get_image_data_1d(tsize, imsize, 0.1)
trials = 1
for trial in range(trials):
    layer1 = Conv1D(filters=1, kernel_size=kernel_size, kernel_regularizer=filter_reg, strides=strides,
                    use_bias=False, activation='relu', batch_size=batch, input_shape=(imsize, 1), padding='same')
    layer2 = MaxPooling1D(pool_size=2)
    layer3 = Dense(units=1, activation='linear', use_bias=False)
    layers_1 = [layer1, layer2, Flatten(), layer3]
    model, err = train_neural_counter(layers_1, data, 'mse')
    print(err)
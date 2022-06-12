from code_for_hw8_keras import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Problem 3A: Uncomment each line to run the best architectures for each validation set
# run_keras_2d("1", archs(2)[0], 10, display=False, verbose=False, trials=5)
# run_keras_2d("2", archs(2)[4], 10, display=False, verbose=False, trials=5)
# run_keras_2d("3", archs(2)[1], 10, display=False, verbose=False, trials=5)
# run_keras_2d("4", archs(2)[1], 10, display=False, verbose=False, trials=5)

# Problem 3B: Uncomment each line to run the best architecture for all validation sets
# run_keras_2d("1", archs(2)[4], 10, display=False, verbose=False, trials=5)
# run_keras_2d("2", archs(2)[4], 10, display=False, verbose=False, trials=5)
# run_keras_2d("3", archs(2)[4], 10, display=False, verbose=False, trials=5)
# run_keras_2d("4", archs(2)[4], 10, display=False, verbose=False, trials=5)

# Problem 3C: Uncoment to train with dataset '3' using architecture (200,200) for 100 epochs and 1 trial
# run_keras_2d("3", [
#     Dense(input_dim=2, units=200, activation="relu"),
#     Dense(units=200, activation="relu"),
#     Dense(units=2, activation="softmax")
# ], 100)

# Problem 3E: Uncomment each of the lines below to get the average validation accuracy for the 3class dataset
# for each of the architectures, using 10 epochs and 5 trials.
# run_keras_2d("3class", archs(3)[0], 10, split=0.5, display=False, verbose=False, trials=5)
# run_keras_2d("3class", archs(3)[1], 10, split=0.5, display=False, verbose=False, trials=5)
# run_keras_2d("3class", archs(3)[2], 10, split=0.5, display=False, verbose=False, trials=5)
# run_keras_2d("3class", archs(3)[3], 10, split=0.5, display=False, verbose=False, trials=5)
# run_keras_2d("3class", archs(3)[4], 10, split=0.5, display=False, verbose=False, trials=5)

# Problem 3G: Compute class predictions
# x, y, model = run_keras_2d("3class", archs(3)[0], 10, split=0.5, display=False, verbose=False, trials=1)
# weights = model.layers[0].get_weights()
# points = np.array([[-1, 0], [1, 0], [0, -11], [0, 1], [-1, -1], [-1, 1], [1, 1], [1, -1]])
# z = weights[0].T @ points.T + weights[1].reshape(-1, 1)
# print(np.argmax(z, axis=0))
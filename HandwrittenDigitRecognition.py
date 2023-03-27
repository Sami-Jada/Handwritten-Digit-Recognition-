import numpy as np
from scipy.io import loadmat
from keras import losses, optimizers
from keras.models import load_model
from keras.layers import Conv2D, Flatten, BatchNormalization 
from keras.layers import Dense, ReLU
from keras.models import Sequential

    # Get image name from user - soon to come
    # Find and Open image in file system - soon to come
    # Predict the number in the image and return it back to the user - soon to come

if __name__ == "__main__":

    # Import data
    data = loadmat("NumberRecognitionBiggest.mat")

    # Extract X_train, X_test, and y_train arrays from imported data
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]

    # Adapted from Francois Chollet at https://keras.io/examples/vision/mnist_convnet/
    # Normalize pixel ranges in X_train and X_test from [0, 255] to [0, 1] and convert to type float (from int)
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Adapted from Francois Chollet at https://keras.io/examples/vision/mnist_convnet/
    # Add channel information using expand_dim for X_train and X_test inputs
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # Transpose y_train array to be compatible with X_train array
    y_train = np.transpose(y_train)

    # Create Convolutional Neural Network to recognize handwritten digits
    CNN = Sequential()
    CNN.add(BatchNormalization())
    CNN.add(
        Conv2D(
        1,
        kernel_size = 3, 
        input_shape = (28, 28, 1), 
        activation="linear",
        data_format="channels_last"
        )
    )
    CNN.add(ReLU())
    CNN.add(Flatten()) # Flatten input's shape before passing to dense layer.
    CNN.add(
        Dense(
        units = 10, 
        activation="softmax") # softmax because multiclass classification output. 10 units because 10 digits in dataset
        )
    
    # Model compilation
    CNN.compile(
       loss=losses.mean_squared_error,
       optimizer=optimizers.SGD(momentum=0.9, learning_rate=0.001),
       metrics=["accuracy"]
    )

    # Train model to recognize handwritten digits
    trained_CNN = CNN.fit(X_train, y_train, batch_size = 8, epochs = 15, verbose=1)

    # Predit handwritten digits in X_test array
    digit_predictions = CNN.predict(X_test)

    # A list to hold the predictions for each image passed to the model.
    list_of_predictions = []

    # for each prediction, we will find the index with the highest probability and we will add that to the list of predictions
    for predictions in digit_predictions:
        list_of_predictions.append(predictions.argmax())

    CNN.save("Convolutional_Neural_Network/HandWrittenDigitRecognition")
    ml_model = load_model('Convolutional_Neural_Network/HandWrittenDigitRecognition')

    
        

    


    





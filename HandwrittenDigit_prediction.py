import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow 
from sklearn.metrics import accuracy_score
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Loading and Analyzing the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
'''print(x_train)
print(y_train)
print(x_test)
print(y_test)
print("Shape of input training dataset is ", x_train.shape)
print("Shape of output training dataset is ", y_train.shape)
print("Shape of input testing dataset is " ,x_test.shape)
print("Shape of output testing dataset is ", y_test.shape)'''

# Displaying the first image in the training set
'''plt.imshow(x_train[40])
plt.title(f'Label: {y_train[40]}')  # Display the label of the image
plt.show()'''

# Normalize the pixel values to be between 0 and 1 for easier training
x_train = x_train / 255.0
x_test = x_test / 255.0
#print(x_train)
#print(x_test)

# Model Training
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten 2D data to 1D
model.add(Dense(128, activation='relu'))  # Hidden layer with 128 nodes
model.add(Dense(32, activation='relu'))   # Additional hidden layer with 32 nodes
model.add(Dense(10, activation='softmax'))  # Output layer with 10 nodes (for 10 classes)

# Compile the model with loss function, optimizer, and evaluation metric
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model with training data and validate on 20% of the data
history = model.fit(x_train, y_train, epochs=4, validation_split=0.2)

# Predict the probability distribution for the test dataset
y_prob = model.predict(x_test)  # Predict probabilities for each class
print(y_prob)

# Convert probabilities to predicted class labels
y_pred = y_prob.argmax(axis=1)  # Get the index of the maximum probability
print(y_pred)

# Calculate the accuracy score of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy percentage:", accuracy*100,"%")

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()  # Add legend to display labels
plt.show()

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()  # Add legend to display labels
plt.show()


# Display the first test image and predict its digit
plt.imshow(x_test[0], cmap='Oranges')
plt.title(f'Predicted Label: {model.predict(x_test[0].reshape(1, 28, 28)).argmax()}')
plt.show()

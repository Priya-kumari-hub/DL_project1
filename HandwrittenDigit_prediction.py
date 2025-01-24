import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist

# Loading and Analyzing the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1 for easier training
x_train = x_train / 255.0
x_test = x_test / 255.0

# Display the first image in the training set (optional)
plt.imshow(x_train[0], cmap='Greys')
plt.title(f'Label: {y_train[0]}')
plt.show()

# Model Definition
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten 2D data to 1D
model.add(Dense(128, activation='relu'))  # Hidden layer with 128 nodes
model.add(Dropout(0.3))  # Dropout with 30% rate to prevent overfitting
model.add(Dense(64, activation='relu'))   # Additional hidden layer with 64 nodes
model.add(Dropout(0.3))  # Dropout with 30% rate
model.add(Dense(10, activation='softmax'))  # Output layer with 10 nodes (for 10 classes)

# Compile the model with loss function, optimizer, and evaluation metric
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with training data and validate on 20% of the data
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict the probability distribution for the test dataset
y_prob = model.predict(x_test)

# Convert probabilities to predicted class labels
y_pred = np.argmax(y_prob, axis=1)

# Calculate and display the accuracy score of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Display a random test image and predict its label
index = 0  # You can change this to display a different test image
plt.imshow(x_test[index], cmap='Oranges')
predicted_label = model.predict(x_test[index].reshape(1, 28, 28)).argmax()
plt.title(f'Predicted Label: {predicted_label}, Actual Label: {y_test[index]}')
plt.show()

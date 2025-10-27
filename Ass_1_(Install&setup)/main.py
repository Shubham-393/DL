import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Check version
print("TensorFlow version:", tf.__version__)

# Simple Sequential Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Dummy data
import numpy as np
x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000,))

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate
loss, acc = model.evaluate(x_train, y_train)
print("Training Accuracy:", acc)


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Function to create CNN model
def create_model():
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu',
                      input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

# Adam Optimizer
adam = create_model()

adam.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

h1 = adam.fit(
    x_train,
    y_train,
    epochs=3,
    verbose=0
)

# RMSProp Optimizer
rms = create_model()

rms.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

h2 = rms.fit(
    x_train,
    y_train,
    epochs=3,
    verbose=0
)

# Evaluation

print("Adam Accuracy:",
      adam.evaluate(x_test, y_test, verbose=0)[1])

print("Adam Loss:",
      h1.history['loss'][0])

print("RMSProp Accuracy:",
      rms.evaluate(x_test, y_test, verbose=0)[1])

print("RMSProp Loss:",
      h2.history['loss'][0])

# Graph for Accuracy
plt.plot(h1.history['accuracy'], label='Adam Accuracy')
plt.plot(h2.history['accuracy'], label='RMSProp Accuracy')

plt.title('Optimizer Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
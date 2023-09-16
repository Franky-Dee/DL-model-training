# Step 1: Import Libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load and Preprocess the CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to between 0 and 1

# Step 3: Define the Attention Mechanism Model
input_shape = (32, 32, 3)

def attention_block(inputs, filters):
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Attention Mechanism
    attention = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(x)
    attention = layers.BatchNormalization()(attention)
    attention = layers.ReLU()(attention)
    attention = layers.Softmax(axis=-1)(attention)
    
    scaled_x = layers.Multiply()([x, attention])
    
    return scaled_x

inputs = keras.Input(shape=input_shape)
x = attention_block(inputs, 32)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = attention_block(x, 64)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = attention_block(x, 128)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
epochs = 10
batch_size = 64

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Step 6: Evaluate the Model
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Step 7: Display Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict(x_test), multi_class='ovr')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("ROC AUC:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Kappa Score:", kappa)

# Optional: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

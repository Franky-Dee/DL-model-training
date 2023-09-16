import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define and train the autoencoder
input_layer = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_test, x_test))

# Create the classifier model by using the encoder part of the autoencoder
classifier_input = Input(shape=(32, 32, 3))
classifier_layer = autoencoder.layers[1](classifier_input)
classifier_layer = autoencoder.layers[2](classifier_layer)
classifier_layer = autoencoder.layers[3](classifier_layer)
classifier_layer = autoencoder.layers[4](classifier_layer)
classifier_layer = Flatten()(classifier_layer)
classifier_layer = Dense(128, activation='relu')(classifier_layer)
classifier_layer = Dense(64, activation='relu')(classifier_layer)
classifier_output = Dense(10, activation='softmax')(classifier_layer)

classifier = Model(classifier_input, classifier_output)

# Compile and train the classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the classifier
y_pred = classifier.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Performance metrics
accuracy = accuracy_score(y_true, y_pred_classes)
confusion = confusion_matrix(y_true, y_pred_classes)
roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
kappa = cohen_kappa_score(y_true, y_pred_classes)

# Print results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')
print(f'AUC/ROC: {roc_auc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Kappa: {kappa}')

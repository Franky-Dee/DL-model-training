import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import LabelBinarizer

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Autoencoder architecture
input_layer = Input(shape=(32, 32, 3))
flatten_layer = Flatten()(input_layer)
encoder_output = Dense(128, activation="relu")(flatten_layer)
decoder_output = Dense(32 * 32 * 3, activation="sigmoid")(encoder_output)
reshape_layer = Reshape((32, 32, 3))(decoder_output)

autoencoder = Model(inputs=input_layer, outputs=reshape_layer)

# Encoder model for feature extraction
encoder = Model(inputs=input_layer, outputs=encoder_output)

# Compile the autoencoder
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# Train the autoencoder
autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)

# Extract features using the encoder
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)

# Convert labels to one-hot encoded format
lb = LabelBinarizer()
y_train_encoded = lb.fit_transform(y_train)
y_test_encoded = lb.transform(y_test)

# Define a classifier (you can use any classifier of your choice)
classifier = tf.keras.Sequential(
    [
        Dense(128, activation="relu", input_shape=(128,)),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the classifier on the encoded features
classifier.fit(
    x_train_encoded,
    y_train_encoded,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_encoded, y_test_encoded),
)

# Predictions
y_pred = classifier.predict(x_test_encoded)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_encoded, axis=1)

# Calculate performance metrics
accuracy = accuracy_score(y_true_labels, y_pred_labels)
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
roc_auc = roc_auc_score(y_test_encoded, y_pred)
precision = precision_score(y_true_labels, y_pred_labels, average="macro")
recall = recall_score(y_true_labels, y_pred_labels, average="macro")
f1 = f1_score(y_true_labels, y_pred_labels, average="macro")
kappa = cohen_kappa_score(y_true_labels, y_pred_labels)

# Display performance metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("ROC AUC:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Kappa:", kappa)

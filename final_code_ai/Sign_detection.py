# SUBMITTED BY THE STUDENTS:         OMAR EHAB ABUDAYYEH             - 2136037
#                                   AHMAD Mohamad AL-HAJ             - 2141147
#                                   MUHAMMAD MUSTAFA AL-MASHAYEKH   - 2138237


import os
import numpy as np
from PIL import Image



class CNN:
    def __init__(self):
        # KERNELS/FILTERS INITIALIZATION
        self.kernels = [
            np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]) / 9,
            np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]) / 9,
            np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]) / 9,
            np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]) / 9,
        ]

        self.num_kernels = len(self.kernels)
        self.nn_input_size = 8 * 8 * self.num_kernels
        self.hidden_layer_size = 64
        self.nn_outputSize = 10

        # WEIGHTS AND BIASES INITIALIZATION
        self.W1 = np.random.randn(self.nn_input_size, self.hidden_layer_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_layer_size))
        self.W2 = np.random.randn(self.hidden_layer_size, self.nn_outputSize) * 0.01
        self.b2 = np.zeros((1, self.nn_outputSize))
        # CONVOLUTION STEP

    def CONVOLVE(self, image, kernel):  # CONVOLUTION LAYER
        h, w = image.shape
        kh, kw = kernel.shape
        conv_output = np.zeros((h - kh + 1, w - kw + 1))
        for i in range(conv_output.shape[0]):
            for j in range(conv_output.shape[1]):
                conv_output[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
        return conv_output

    def MAXPOOLING(self, feature_map, pool_size=2):  # POOL LAYER
        height, width = feature_map.shape
        pooled_height = height // pool_size
        pooled_width = width // pool_size

        pooled_map = np.zeros((pooled_height, pooled_width))

        for i in range(pooled_height):
            for j in range(pooled_width):
                start_i = i * pool_size
                start_j = j * pool_size
                pooled_map[i, j] = np.max(feature_map[start_i:start_i + pool_size, start_j:start_j + pool_size])

        return pooled_map

    def RELU(self, x):
        return np.maximum(0, x)

    def SOFTMAX(self, x):  # SOFTMAX FOR CLASSIFICATION
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def FORWARD_PASS(self, image):  # FORWARD_PASS
        self.cache = {}
        conv_outputs = [np.maximum(0, self.CONVOLVE(image, k)) for k in self.kernels]
        self.cache['CONVOLUTION_OUTPUTS'] = conv_outputs

        pooled_outputs = [self.MAXPOOLING(c) for c in conv_outputs]
        self.cache['POOLED_OUTPUTS'] = pooled_outputs

        flattened = np.concatenate([p.flatten() for p in pooled_outputs])
        self.cache['FLATTENED'] = flattened

        z1 = flattened.dot(self.W1) + self.b1
        a1 = self.RELU(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1

        z2 = a1.dot(self.W2) + self.b2
        a2 = self.SOFTMAX(z2)
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        return a2

    def BACK_PROPAGATION(self, X, y, LearningRate=0.01):  # BACKWARD PROPAGATION, UPDATING LOSS AND WEIGHTS
        m = X.shape[0]
        a2 = self.cache['a2']
        a1 = self.cache['a1']
        flattened = self.cache['FLATTENED']

        dz2 = a2.copy()
        dz2[range(m), y] -= 1
        dz2 /= m
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = dz2.dot(self.W2.T) * (a1 > 0)
        dW1 = flattened.reshape(m, -1).T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= (LearningRate * dW2)
        self.b2 -= (LearningRate * db2)
        self.W1 -= (LearningRate * dW1)
        self.b1 -= (LearningRate * db1)

    def PREDICT(self, image):
        probs = self.FORWARD_PASS(image)
        return np.argmax(probs)

    def TRAIN(self, X_train, y_train, epochs=200, LearningRate=0.01, batch_size=32):
        m = X_train.shape[0]
        losses, accuracies = [], []
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_train = X_train[indices]
            y_train = y_train[indices]

            batch_losses, batch_accuracies = [], []
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                for j in range(len(X_batch)):
                    pred = self.FORWARD_PASS(X_batch[j])
                    loss = -np.log(pred[0, y_batch[j]] + 1e-8)
                    batch_losses.append(loss)
                    acc = int(np.argmax(pred) == y_batch[j])
                    batch_accuracies.append(acc)
                    self.BACK_PROPAGATION(X_batch[j].reshape(1, -1), np.array([y_batch[j]]), LearningRate
                                          )

            losses.append(np.mean(batch_losses))
            accuracies.append(np.mean(batch_accuracies))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")

        return losses, accuracies


def LOAD_IMAGES(TRAINING_DIRECTORY):  # LOADING IMAGES
    images, labels = [], []
    for sign_number in range(10):
        for i in range(1, 10000):
            img_name = f"{sign_number}_{i}.PNG"
            img_path = os.path.join(TRAINING_DIRECTORY, img_name)
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert('L').resize((19, 19))
                img_array = np.array(img) / 255.0
                img_array = (img_array < 0.5).astype(np.float32)
                images.append(img_array)
                labels.append(sign_number)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)


def TRAIN_CNN():
    TRAINING_DIRECTORY = "C:/Users/omar_/OneDrive/Desktop/;;/AI & ML/AI Project/TrainingImages/justatest5"
    print("Loading and preprocessing training images..")
    images, labels = LOAD_IMAGES(TRAINING_DIRECTORY)

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(images))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = images[train_idx], labels[train_idx]
    X_test, y_test = images[test_idx], labels[test_idx]

    print("Creating and training CNN...")
    cnn = CNN()
    losses, accuracies = cnn.TRAIN(X_train, y_train, epochs=50, LearningRate=0.05)

    print("Evaluating on test set...")
    correct = sum(cnn.PREDICT(X_test[i]) == y_test[i] for i in range(len(X_test)))
    test_accuracy = correct / len(X_test)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    return cnn, losses, accuracies, test_accuracy


def TEST_CNN(model_path, TESTING_DIRECTORY):  # Testing the trained CNN

    print("Loading trained model weights...")
    cnn = CNN()
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    data = np.load(model_path)
    cnn.W1 = data["W1"]
    cnn.b1 = data["b1"]
    cnn.W2 = data["W2"]
    cnn.b2 = data["b2"]

    print("Loading and preprocessing test data...")
    test_images, test_labels = LOAD_IMAGES(TESTING_DIRECTORY)
    if len(test_images) == 0:
        print("No valid images found in test directory.")
        return

    print("Running predictions...")
    correct = 0
    for i in range(len(test_images)):
        pred = cnn.PREDICT(test_images[i])
        true = test_labels[i]
        print(f"Image {i + 1}: Predicted = {pred}, Actual = {true}")
        if pred == true:
            correct += 1

    accuracy = correct / len(test_images)
    print(f"Testing Accuracy on {len(test_images)} images: {accuracy:.4f}")


if __name__ == "__main__":
    #   ------------------------------------------------ TEST BELOW ------------------------------------------------ #

    # *****Weights are loaded inside "TEST_CNN()" function.*****

    current_directory = os.getcwd()
    images_directory = current_directory+'/../GRADING_IMAGES/'
    TEST_CNN("parameters.npz", images_directory)

#   ------------------------------------------------ TRAIN BELOW ------------------------------------------------ #


# cnn, losses, accuracies, test_accuracy = TRAIN_CNN()
# np.savez("parameters.npz", W1=cnn.W1, b1=cnn.b1, W2=cnn.W2, b2=cnn.b2)

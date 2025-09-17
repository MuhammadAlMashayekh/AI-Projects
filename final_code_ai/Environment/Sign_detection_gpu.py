import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Determine if CUDA (GPU) is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNN_PyTorch(nn.Module):
    def __init__(self):
        super(CNN_PyTorch, self).__init__()

        # Kernels (as numpy arrays initially)
        # These kernels are fixed and will not be trained.
        self.numpy_kernels = [
            np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]) / 9,
            np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]) / 9,
            np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]) / 9,
            np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]) / 9,
        ]
        self.num_kernels = len(self.numpy_kernels)

        # Create a list of Conv2d layers, one for each fixed kernel.
        # Each Conv2d layer will have 1 input channel and 1 output channel.
        # The weights of these layers are set from numpy_kernels and are not trainable.
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_kernels):
            kernel_numpy = self.numpy_kernels[i]
            kh, kw = kernel_numpy.shape
            conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kh, kw), padding=0, bias=False)

            # Set the kernel weights. Conv2d weights need to be in the shape:
            # (out_channels, in_channels, kernel_height, kernel_width)
            conv.weight.data = torch.tensor(kernel_numpy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            conv.weight.requires_grad = False # Make these kernels non-trainable (fixed)
            self.conv_layers.append(conv)

        # Max pooling layer (2x2 kernel with a stride of 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened features after convolution and pooling.
        # Input image size is 19x19 (as per load_and_preprocess_data resize).
        # After 3x3 convolution (no padding): 19 - 3 + 1 = 17. Output feature map: 17x17.
        # After 2x2 max pooling (stride 2): 17 // 2 = 8. Output feature map: 8x8.
        self.feature_map_size_after_pool = 8
        self.nn_input_size = self.feature_map_size_after_pool * self.feature_map_size_after_pool * self.num_kernels

        self.hidden_layer_size = 64
        self.output_size = 10 # For digits 0-9

        # Fully connected layers. These layers will be trained.
        self.fc1 = nn.Linear(self.nn_input_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_size)

        # Move the entire model to the designated device (GPU or CPU)
        self.to(device)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor. Expected shapes:
                                - Single image: (H, W)
                                - Batch of images: (N, H, W)
                                - Batch of images with channel: (N, C_in, H, W) where C_in=1
        Returns:
            torch.Tensor: Raw logits from the output layer (N, num_classes).
        """
        # Ensure input tensor x has the correct shape (N, C_in, H, W)
        if x.ndim == 2: # Single image (H,W)
            x = x.unsqueeze(0).unsqueeze(0) # Add batch and channel dimensions: (1, 1, H, W)
        elif x.ndim == 3: # Batch of images (N, H, W)
            x = x.unsqueeze(1) # Add channel dimension: (N, 1, H, W)

        x = x.to(device) # Ensure input tensor is on the correct device

        # Apply each fixed convolution layer, followed by ReLU activation
        conv_relu_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_out_relu = F.relu(conv_out)
            conv_relu_outputs.append(conv_out_relu)

        # Apply max pooling to each of the feature maps
        pooled_outputs = [self.pool(c_out) for c_out in conv_relu_outputs]

        # Concatenate the pooled feature maps along the channel dimension (dim=1).
        # Each pooled_output is (N, 1, 8, 8).
        # After concatenation, `flattened` becomes (N, num_kernels, 8, 8).
        flattened = torch.cat(pooled_outputs, dim=1)

        # Flatten the tensor for the fully connected layers.
        # Reshape from (N, num_kernels, 8, 8) to (N, num_kernels * 8 * 8).
        flattened = flattened.view(flattened.size(0), -1)

        # Pass through the fully connected layers
        z1 = self.fc1(flattened)
        a1 = F.relu(z1)
        logits = self.fc2(a1) # Output raw logits (pre-softmax)

        return logits

    def predict_proba(self, image_tensor):
        """
        Predicts class probabilities for a single image.
        Args:
            image_tensor (torch.Tensor): A single image tensor (H, W) or (1, H, W).
        Returns:
            torch.Tensor: Probabilities for each class (1, num_classes).
        """
        self.eval() # Set the model to evaluation mode
        with torch.no_grad(): # Disable gradient calculations
            logits = self.forward(image_tensor) # Forward pass handles unsqueezing and device transfer
            probabilities = F.softmax(logits, dim=1)
            return probabilities

    def predict(self, image_tensor):
        """
        Predicts the class index for a single image.
        Args:
            image_tensor (torch.Tensor): A single image tensor (H, W) or (1, H, W).
        Returns:
            int: The predicted class index.
        """
        self.eval() # Set the model to evaluation mode
        with torch.no_grad(): # Disable gradient calculations
            logits = self.forward(image_tensor) # Forward pass handles unsqueezing and device transfer
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            return predicted_class.item()


def load_and_preprocess_data_pytorch(data_dir):
    """
    Load images from directory, preprocess them, and convert to PyTorch tensors.
    Assumes image names like 0_1.PNG, 0_2.PNG, ..., 9_N.PNG.
    Args:
        data_dir (str): Path to the directory containing images.
    Returns:
        tuple: (images_tensor, labels_tensor)
               - images_tensor: A tensor of shape (num_images, height, width).
               - labels_tensor: A tensor of shape (num_images,).
    """
    images_list, labels_list = [], []
    print(f"Loading images from: {data_dir}")
    for sign_number in range(10):  # Classes 0 to 9
        file_count = 0
        # Increased range slightly to be safer if initial numbers are skipped
        for i in range(1, 10001):
            img_name = f"{sign_number}_{i}.PNG"
            img_path = os.path.join(data_dir, img_name)
            if not os.path.exists(img_path):
                # Heuristic to break early if no files found for a class after a reasonable count
                if i > 250 and file_count == 0:
                    break
                continue

            try:
                img = Image.open(img_path).convert('L').resize((19, 19))
                img_array = np.array(img) / 255.0
                img_array_binarized = (img_array < 0.5).astype(np.float32) # Binarize

                images_list.append(torch.tensor(img_array_binarized, dtype=torch.float32))
                labels_list.append(torch.tensor(sign_number, dtype=torch.long))
                file_count += 1
            except Exception as e:
                print(f"Error loading or processing {img_path}: {e}")

    if not images_list:
        print("No images were successfully loaded.")
        return torch.empty(0), torch.empty(0)

    images_tensor = torch.stack(images_list)
    labels_tensor = torch.stack(labels_list)
    print(f"Loaded {images_tensor.shape[0]} images.")
    return images_tensor, labels_tensor


def train_cnn_pytorch(model, X_train, y_train, epochs=50, learning_rate=0.005, batch_size=32):
    """
    Trains the PyTorch CNN model.
    Args:
        model (CNN_PyTorch): The model to train.
        X_train (torch.Tensor): Training images (N, H, W).
        y_train (torch.Tensor): Training labels (N,).
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
    Returns:
        tuple: (losses_history, accuracies_history) lists of epoch-wise metrics.
    """
    criterion = nn.CrossEntropyLoss()
    # Only pass parameters that require gradients to the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    num_samples = X_train.size(0)
    losses_history, accuracies_history = [], []

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train() # Set the model to training mode

        # Shuffle data at the beginning of each epoch
        permutation = torch.randperm(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        current_epoch_loss = 0.0
        current_epoch_correct_predictions = 0

        for i in range(0, num_samples, batch_size):
            # Get batch
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size].to(device) # Move labels to device

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get logits from the model
            # Model's forward pass handles device transfer for X_batch
            logits = model(X_batch)

            # Calculate loss
            loss = criterion(logits, y_batch)

            # Backward pass: Compute gradients
            loss.backward()

            # Optimize: Update model parameters
            optimizer.step()

            # Accumulate loss for the epoch
            current_epoch_loss += loss.item() * X_batch.size(0) # loss.item() is avg loss for batch

            # Calculate accuracy for the batch (useful for tracking training progress)
            _, predicted_classes = torch.max(logits, 1) # Get class with highest logit
            current_epoch_correct_predictions += (predicted_classes == y_batch).sum().item()

        avg_epoch_loss = current_epoch_loss / num_samples
        avg_epoch_accuracy = current_epoch_correct_predictions / num_samples

        losses_history.append(avg_epoch_loss)
        accuracies_history.append(avg_epoch_accuracy)

        # Print progress every few epochs or at the end
        if epoch % 10 == 0 or epoch == epochs - 1:
             print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Training Accuracy: {avg_epoch_accuracy:.4f}")

    return losses_history, accuracies_history # Return training history


def evaluate_cnn_pytorch(model, X_test, y_test, batch_size=32):
    """
    Evaluates the PyTorch CNN model on the test set.
    Args:
        model (CNN_PyTorch): The trained model.
        X_test (torch.Tensor): Test images (N, H, W).
        y_test (torch.Tensor): Test labels (N,).
        batch_size (int): Batch size for evaluation.
    Returns:
        float: Test accuracy.
    """
    # This function remains here but won't be called by the modified training flow
    model.eval() # Set the model to evaluation mode
    total_correct_predictions = 0
    total_samples_processed = 0

    with torch.no_grad(): # Disable gradient calculations during evaluation
        num_samples = X_test.size(0)
        for i in range(0, num_samples, batch_size):
            X_batch = X_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size].to(device) # Move labels to device

            # Model's forward pass handles device transfer for X_batch
            logits = model(X_batch)

            _, predicted_classes = torch.max(logits, 1) # Get class with highest logit
            total_correct_predictions += (predicted_classes == y_batch).sum().item()
            total_samples_processed += y_batch.size(0)

    accuracy = total_correct_predictions / total_samples_processed if total_samples_processed > 0 else 0.0
    print(f"Test set accuracy: {accuracy:.4f} ({total_correct_predictions}/{total_samples_processed})")
    return accuracy

def run_training_only(TESTING_DIRECTORY): # Renamed for clarity
    """
    Main function to orchestrate data loading, model training, and saving.
    Skips the final test set evaluation.
    """
    # --- IMPORTANT: Update this path to your actual data directory ---
    # This directory should contain your training images
    # Example for relative path: data_dir = "./TrainingImgs"
    data_dir = TESTING_DIRECTORY # Use your training data directory

    print("Step 1: Loading and preprocessing data for PyTorch...")
    images, labels = load_and_preprocess_data_pytorch(data_dir)

    if images.numel() == 0: # Check if any images were loaded
        print(f"No images loaded from {data_dir}. Cannot proceed with training.")
        print("Please check that the 'data_dir' path is correct and contains valid images.")
        return None, [], [] # Return empty values

    # Shuffle and split data into training and (unused for final eval) testing sets
    num_images = images.size(0)
    indices = torch.randperm(num_images)
    # Use most data for training if no separate validation is needed
    # Or adjust split_idx if you *do* want a separate validation set within train_cnn_pytorch
    split_idx = int(0.8 * num_images) # Still split, e.g., 80% train, 20% ignored for final eval

    # --- FIX: ADD THE MISSING LINE TO DEFINE train_idx and test_idx ---
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    # --- END FIX ---

    X_train, y_train = images[train_idx], labels[train_idx]
    X_test, y_test = images[test_idx], labels[test_idx] # Test set loaded but not used for final eval

    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples (not used for final eval).")

    print("\nStep 2: Creating and training CNN with PyTorch...")
    cnn_model = CNN_PyTorch() # Model is automatically moved to `device` in its __init__

    # Train the model
    # train_cnn_pytorch will return training history, but no test accuracy
    train_losses, train_accuracies = train_cnn_pytorch(cnn_model, X_train, y_train,
                                                        epochs=30, learning_rate=0.005, batch_size=32)

    print("\nStep 3: Training complete.")
    # We skip the final test evaluation call here.

    return cnn_model, train_losses, train_accuracies # Return the trained model and training history

# ... (rest of the code including predict_with_saved_model_on_directory and __main__ block)
# The predict function remains the same but will be commented out in __main__
def predict_with_saved_model_on_directory(model_path, image_data_dir):
     # ... (this function remains unchanged from your original code)
     print(f"Loading trained PyTorch model from: {model_path}")
     cnn_model = CNN_PyTorch() # Initialize model structure

     if not os.path.exists(model_path):
         print(f"Model file not found: {model_path}")
         print("Ensure the model has been trained and saved, or the path is correct.")
         return

     try:
         # Load the saved state dictionary into the model.
         # map_location ensures model loads correctly regardless of where it was trained (CPU/GPU).
         cnn_model.load_state_dict(torch.load(model_path, map_location=device))
         cnn_model.to(device) # Ensure model is on the correct device after loading
         cnn_model.eval() # Set to evaluation mode
         print("Model loaded successfully.")
     except Exception as e:
         print(f"Error loading model weights from {model_path}: {e}")
         return

     print(f"\nLoading and preprocessing image data for prediction from: {image_data_dir}")
     # The load function expects labels, but for prediction on new data, labels might not be known
     # or might be used for comparison if available.
     # If your prediction images don't follow the "class_idx.PNG" naming,
     # you might need a different loading function for pure prediction.
     # For now, we assume the same structure for simplicity of demonstration.
     images_for_prediction, true_labels_if_available = load_and_preprocess_data_pytorch(image_data_dir)

     if images_for_prediction.numel() == 0:
         print(f"No valid images found in the directory: {image_data_dir}")
         return

     num_images_to_predict = images_for_prediction.size(0)
     print(f"\nRunning predictions on {num_images_to_predict} images from '{image_data_dir}'...")

     correct_predictions = 0

     with torch.no_grad(): # Disable gradient calculations for inference
         for i in range(num_images_to_predict):
             # Get a single image
             image_tensor = images_for_prediction[i] # Shape: (H, W)

             # The model's `predict` method handles unsqueezing, device transfer, and prediction.
             predicted_label = cnn_model.predict(image_tensor)

             actual_label_info = ""
             if true_labels_if_available.numel() > 0 : # Check if labels were loaded
                 actual_label = true_labels_if_available[i].item()
                 actual_label_info = f", Actual = {actual_label}"
                 if predicted_label == actual_label:
                     correct_predictions += 1

             print(f"Image {i+1}/{num_images_to_predict}: Predicted = {predicted_label}{actual_label_info}")

     if true_labels_if_available.numel() > 0 and num_images_to_predict > 0:
         accuracy = correct_predictions / num_images_to_predict
         print(f"\nPrediction Accuracy on directory '{image_data_dir}': {accuracy:.4f} ({correct_predictions}/{num_images_to_predict})")
     elif num_images_to_predict > 0:
         print(f"\nCompleted predictions for {num_images_to_predict} images.")
     else:
         print("No samples were available to predict in the directory.")


def train_and_save_model(Train_DIRECTORY):
    print("--- Starting Training Process ---")
    trained_model, train_losses, train_accuracies = run_training_only(Train_DIRECTORY)

    if trained_model:
        model_save_path = "parameters_gpu.npz"
        try:
            torch.save(trained_model.state_dict(), model_save_path)
            print(f"\nTrained PyTorch model weights saved to {model_save_path}")
            if train_accuracies:
                print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
        except Exception as e:
            print(f"Error saving model to {model_save_path}: {e}")
    else:
        print("Training was not completed or failed.")
def run_prediction(image_data_dir): # Renamed the function
    """
    Loads a pre-trained model and runs predictions on images in a specified directory.
    Args:
        image_data_dir (str): The path to the directory containing images for prediction.
                              Assumes image filenames follow the pattern 'digit_index.PNG'.
    """
    print("\n--- Starting Prediction Process with a Pre-trained Model ---")

    image_dir_for_prediction = image_data_dir
    path_to_model_for_prediction = "parameters_gpu.npz"

    if not os.path.exists(path_to_model_for_prediction):
        print(f"Pretrained model '{path_to_model_for_prediction}' not found.")
        print("Please train and save a model first by uncommenting and running Option 1,")
        print("or ensure the path to an existing .pth file is correct.")
    else:
        if not os.path.isdir(image_dir_for_prediction):
            print(f"Image directory for prediction '{image_dir_for_prediction}' not found. Please check the path.")
        else:
            predict_with_saved_model_on_directory(path_to_model_for_prediction, image_dir_for_prediction)

    print("\nPrediction process finished.") # Add a concluding print
# ... (predict_with_saved_model_on_directory, train_and_save_model)

# Main execution block
if __name__ == "__main__":
    # This block runs ONLY when you execute the script directly using 'python Training_gpu2.py'

    print("Script started.")

    # --- Call the test_function here ---

    # 1. Define the path to the directory containing your testing images
    #    **IMPORTANT:** Replace this path with the actual path on your computer
    path_to_your_test_images = "C:/Users/ADMIN/Downloads/justatest5 (1)/justatest5"

    print(f"Calling test_function with directory: {path_to_your_test_images}")

    # 2. Call the test_function and pass the path to it
    # run_prediction(path_to_your_test_images)
    train_and_save_model(path_to_your_test_images)
    print("Script finished.")

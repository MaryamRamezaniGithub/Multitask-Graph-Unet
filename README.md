# Image-Segmentation

This project implements a human segmentation model using PyTorch and the segmentation_models_pytorch library. The goal is to segment human figures from images, and the code is organized into several tasks.

Setup and Dependencies
Ensure you have the necessary dependencies installed before running the code. You can install them using the following:

pip install torch opencv-python pandas matplotlib scikit-learn tqdm albumentations segmentation-models-pytorch

Project Structure
Setup and Configuration:

Import required libraries and set up configurations such as file paths, device (CPU or CUDA), training epochs, learning rate, batch size, image size, encoder type, and pre-trained weights.
Data Loading:

Load the dataset from a CSV file containing image and mask file paths. Split the dataset into training and validation sets.
Data Augmentation:

Define functions to get data augmentation pipelines for the training and validation datasets.
Custom Dataset:

Create a custom PyTorch dataset (SegmentationDataset) to load and preprocess images and masks.
Dataset Exploration:

Display sample images and masks to visually inspect the dataset.
Data Loader:

Load the dataset into PyTorch data loaders for training and validation.
Segmentation Model:

Define a custom segmentation model class (SegmentationModel) using the segmentation_models_pytorch library.
Training Functions:

Implement training and evaluation functions to train the segmentation model.
Training Loop:

Train the model over multiple epochs, saving the best model based on validation loss.
Inference:

Load the trained model and perform inference on a sample image from the validation set.
Visualization:

Display the original image, ground truth mask, and predicted mask for visual inspection.
Instructions

Notes
Customize hyperparameters, model architecture, or training parameters based on your requirements.
Adjust data augmentation techniques in the get_train_augs and get_valid_augs functions.
Explore different encoders and pre-trained weights for the segmentation model.
Extend the code for testing on additional images or integrating with deployment frameworks.

# Convolutional Neural Network (CNN) for MNIST Classification
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model is trained to recognize numbers from 0 to 9 based on the pixel values of 28x28 grayscale images.

## Requirements

To run this project, you need the following libraries:

- PyTorch
- torchvision
- matplotlib
- numpy

You can install the required packages using pip:

```bash
``` pip install torch torchvision matplotlib numpy ```


## Training Approach 

Pull the training and test data from the dataset and convert data to tensors 

Training dataset consists of 60,000 28x28 pixels images with no RGB color values associated with the images 

Testing dataset consists of 10,000 28x28 pixels images also with no RGB color values associated with the images 

Images classes consist of the images with the numbers 1-9  

## Model Architecture

The CNN model consists of:

- Two convolutional layers with ReLU activations and max pooling.
- A dropout layer for regularization.
- Two fully connected layers leading to the output layer with softmax activation.

## Training

The model is trained for 10 epochs with a batch size of 100. During training, the average loss and accuracy are computed for both the training and test sets.

### Training Loop

The training process includes the following steps:

1. **Forward pass**: Compute the model's predictions and calculate the loss.
2. **Backward pass**: Compute gradients and update model weights using the Adam optimizer.
3. **Evaluation**: The model is evaluated on the test set after each training epoch.

## Visualization

The project provides visualizations of training and test loss and accuracy over epochs using Matplotlib. After training, the script displays the following:

- Loss curves for training and testing.
- Accuracy curves for training and testing.
- A sample image from the test set with the model's prediction.

## Example Output

After running the script, you will see the training progress in the console, including loss and accuracy metrics for each epoch. At the end, a sample prediction is displayed along with the corresponding image.

## Conclusion

This project demonstrates the application of a CNN for image classification tasks using the MNIST dataset. The implemented model can be further optimized and modified for improved performance or adapted for other datasets.

After training the model over 11 epochs, the model reulted in having a maximum accuracy of 98% and training accuracy of about 93%. The overall loss during training was around 0.0155 over 10 epochs. Testing overall loss was less than 0.0150 over 10 epochs


## Acknowledgements

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)




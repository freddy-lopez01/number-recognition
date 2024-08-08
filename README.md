# number-recognition
pytorch ml model trained to recognize hand written numbers from the torchvision MNIST datasets  


## Training Approach 

Pull the training and test data from the dataset and convert data to tensors 

Training dataset consists of 60,000 28x28 pixels images with no RGB color values associated with the images 

Testing dataset consists of 10,000 28x28 pixels images also with no RGB color values associated with the images 

Images classes consist of the images with the numbers 1-9  

# Results 

After training the model over 11 epochs, the model reulted in having a maximum accuracy of 98% and training accuracy of about 93%. The overall loss during training was around 0.0155 over 10 epochs. Testing overall loss was less than 0.0150 over 10 epochs



from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch 
import numpy
import matplotlib.pyplot as plt
import logging 

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # create a 2-layer conveluted network 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # (# of channels, # of channels out, kernel size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # (# of channels, # of channels out, kernel size)
        # Dropout 2D layer (regularization layer)
        self.conv2_drop = nn.Dropout2d() # (# of channels, # of channels out, kernel size)
        #dense layer
        self.fc1 = nn.Linear(320, 50) # 350 in and 50 out 
        self.fc2 = nn.Linear(50, 10) # 50 in and 10 out 

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flatten all the data
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)
        

def train(epoch):
    model.train() # set the model into training mode
    correct = 0
    total_loss = 0
    for batch_index, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device) # make sure all the data is going to the same device
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()


        curr_batch = batch_index * len(data)
        overall_size = len(loaders['train'].dataset)
        curr_frac = 100. * curr_batch / overall_size


        if batch_index % 20 == 0:
            print(f"train epoch: {epoch} [{curr_batch}/{overall_size}  ({curr_frac:.1f}%)]\t{loss.item():.6f}")

    avg_loss = total_loss / len(loaders['train'].dataset)
    accuracy = 100. * correct / len(loaders['train'].dataset)
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Train Epoch: {epoch} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    accuracy = 100. * correct / len(loaders['test'].dataset)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({accuracy:.0f}%\n)")


train_data = datasets.MNIST(
    root = 'data', 
    train = True,
    transform = ToTensor(), 
    download = True
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor(), 
    download = True
)
    # Print the amount of samples contained 
print(train_data.targets.size())
    # Print the number of samples (images) and the dimensions (in pixels) of each image
    # After printing out the data (black and white images, no RGB values)
print(train_data.data.size()) # 60,000 images with 28x28 pixesl
print(test_data.data.size()) # 10,000 images with 28x28 pixesl

print(train_data.targets) # shows individual classes of data (numbers 1-9)


loaders = {
        'train': DataLoader(train_data, 
                            batch_size = 100, 
                            shuffle = True, 
                            num_workers = 1), 
        'test': DataLoader(test_data, 
                            batch_size = 100, 
                            shuffle = True, 
                            num_workers = 1), 
}

print(loaders)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) # keep the learning rate low

loss_fn = nn.CrossEntropyLoss()


if __name__ == "__main__":

    for epoch in range(1, 11):
        train(epoch)
        test()

    epochs = range(1, 11)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, test_losses, 'b', label='Test loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g', label='Training accuracy')
    plt.plot(epochs, test_accuracies, 'b', label='Test accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

    model.eval()

    data, target = test_data[4]

    data = data.unsqueeze(0).to(device)

    output = model(data)

    prediction = output.argmax(dim=1, keepdim=True).item()

    print(f"Prediction: {prediction}")

    image = data.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.show()










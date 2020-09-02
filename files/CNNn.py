import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms
#from torchsummary import summary


import time
import os
import glob
from pathlib import Path
DATA_DIR = Path('data')
input_size = (128, 128)
batch_size = 32
num_workers = 4


# 3 sets: train, validation and test sets
# compose chains our sets together
#toTensor transforms the pixel array of our images to a tensor of binary values
#finally, normalize each varient to have a mean between 0 and 1
data_transforms = {
    'Train': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          #transforms.Grayscale(3),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ]),
    'Validation': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          #transforms.Grayscale(3),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ]),
    'Test': transforms.Compose([transforms.Resize(input_size),
                               transforms.ToTensor(),
                              # transforms.Grayscale(3),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ])
}



image_datasets = {x: ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['Train', 'Validation']}

# dataset class to load images with no labels, for our testing set to submit to the competition
class ImageLoader(Dataset):
    def __init__(self, root, transform=None):
        # get image file paths
        self.images = sorted(glob.glob(os.path.join(root, "*")), key=self.glob_format),
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            return img
        else:
            return transforms.ToTensor(img)
         
    @staticmethod
    def glob_format(key):   
        #parses and shortens image filename to int
        key = key.split("/")[-1].split(".")[0]
        key = os.path.basename(key)
        return "{:04d}".format(int(key))

#image_datasets['Test'] = data_transforms['Test'](transforms.ToPILImage()(image_datasets["Test"]))
image_datasets['Test'] = ImageLoader(str(DATA_DIR / "Test"), transform=data_transforms["Test"])

dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory=True)
              for x in ['Train', 'Validation']}

test_loader = DataLoader(dataset = image_datasets['Test'], batch_size = 1, shuffle=False)

gauge_ranges = image_datasets['Train'].classes
#print('classes: ', gauge_ranges)

# prints number of images in each dataset we created
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}

##print('Train Length: {} | Valid Length: {} | Test Length: {}'.format(dataset_sizes['Train'], 
##                                                                     dataset_sizes['Validation'], dataset_sizes['Test']))


# We want to use the GPU if available, if not we use the CPU
# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device
# pylint: enable=E1101

'''
# Plots a given number of images from a PyTorch Data
def show_random_imgs(num_imgs):
  
    for i in range(num_imgs):
        # We're plotting images from the training set
        train_dataset = image_datasets['Train']
        
        # Choose a random image
        rand = np.random.randint(0, len(train_dataset) + 1)
        
        # Read in the image
        ex = img.imread(train_dataset.imgs[rand][0])
        
        # Get the image's label
        percent = (20 * int(gauge_ranges[train_dataset.imgs[rand][1]])) 
        if percent == 0:
            percentile = '(' + str(percent) + ')th Percentile'
        else:
            percentile = '(' +str(percent-20) + ' - ' + str(percent) + ')th Percentile'
        
        # Show the image and print out the image's size (really the shape of it's array of pixels)
        plt.imshow(ex)
        print('Image Shape: ' + str(ex.shape))
        plt.axis('off')
        plt.title(percentile)
        plt.show()
       
  # Plots a batch of images served up by PyTorch    
def show_batch(batch):
  
    # Undo the transformations applied to the images when loading a batch
    batch = batch.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    batch = std * batch + mean
    batch = np.clip(batch, 0, 1)
    
    # Plot the batch
    plt.axis('off')
    plt.imshow(batch)
    
    # pause a bit so that plots are updated
    plt.pause(0.001)
# show_random_imgs(2)
'''
#Get a batch of training data (32 random images)
#imgs, classes = next(iter(dataloaders['Train']))

#makes a grid of images from a batch for us
#batch = torchvision.utils.make_grid(imgs)

#show_batch(batch)

#Calculates the padding size neccessary to create an output of desired dimensions

def get_padding(input_dim, output_dim, kernel_size, stride):
  # Calculates padding necessary to create a certain output size,
  # given a input size, kernel size and stride
  
    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2
    if padding < 0:
        return 0
    else:
        return padding

    # Make sure you calculate the padding amount needed to maintain the spatial size of the input

# after each Conv layer

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,5) #first convolutional layer
        self.pool = nn.MaxPool2d(2,2) # pooling layer
        self.conv2 = nn.Conv2d(16,32,5) # 2nd convoltion
        self.fc1 = nn.Linear(32*29*29,720)
        self.fc2 = nn.Linear(720, 504)
        self.fc3 = nn.Linear(504, 51)


        # nn.Sequential() is simply a container that groups layers into one object
        # Pass layers into it separated by commas
        #self.block1 = nn.Sequential(
          #  nn.Conv2d(3,32,5)
            # The first convolutional layer. Think about how many channels the input starts off with
            # Let's have this first layer extract 32 features
            # YOUR CODE HERE
           # raise NotImplementedError()
            
            # Don't forget to apply a non-linearity
            # YOUR CODE HERE
          #  nn.Relu(x)
            #raise NotImplementedError()
        #)
        
        
        # Applying a global pooling layer
        # Turns the 128 channel rank 4 tensor into a rank 2 tensor of size 32 x 128 (32 128-length
        #  arrays, one for each of the inputs in a batch)
        #self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        #self.fc1 = nn.Linear(128, 512)
        
        # Introduce dropout to reduce overfitting
        #self.drop_out = nn.Dropout(0.5)
        
        # Final fully connected layer creates the prediction array
        #self.fc2 = nn.Linear(512, len(gauge_ranges))
    
    # Feed the input through each of the layers we defined 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,32*29*29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)


        # Input size changes from (32 x 3 x 224 x 224) to (32 x 32 x 224 x 224)
        #x = self.block1(x)
        
        # Size changes from (32 x 32 x 224 x 224) to (32 x 64 x 112 x 112) after max pooling
        #x = self.block2(x)
        
        # Size changes from (32 x 64 x 112 x 112) to (32 x 128 x 56 x 56) after max pooling
        #x = self.block3(x)
        
        # Reshapes the input from (32 x 128 x 56 x 56) to (32 x 128)
        #x = self.global_pool(x)
        #x = x.view(x.size(0), -1)
        
        # Fully connected layer, size changes from (32 x 128) to (32 x 512)
       # x = self.fc1(x)
       # x = self.drop_out(x)
        
        # Size change from (32 x 512) to (32 x 133) to create prediction arrays for each of the images in the batch
        #x = self.fc2(x)
        
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
epochs = 2
model.to(device)
##summary(model, (3,128,128))

def run_epoch(epoch, model, optimizer, dataloaders, device, phase):
  
    running_loss = 0.0
    running_corrects = 0

    if phase == 'Train':
        model.train()
    else:
        model.eval()

    # Looping through batches
    
    for i, (inputs, labels) in enumerate(dataloaders[phase]):
        # ensures we're doing this calculation on our GPU if possible
        #print(input_size)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero parameter gradients
        optimizer.zero_grad()
    
        # Calculate gradients only if we're in the training phase
        with torch.set_grad_enabled(phase == 'Train'):
      
            # This calls the forward() function on a batch of inputs
            outputs = model(inputs)

            # Calculate the loss of the batch

            loss = criterion(outputs, labels)

            # Gets the predictions of the inputs (highest value in the array)
            # pylint: disable=E1101
            _, preds = torch.max(outputs, 1)
            # pylint: enable=E1101
            # Adjust weights through backpropagation if we're in training phase
            if phase == 'Train':
                loss.backward()
                optimizer.step()

        # Document statistics for the batch
        running_loss += loss.item() * inputs.size(0)
        #pylint: disable=E1101
        running_corrects += torch.sum(preds == labels.data)
        # pylint: enable=E1101
    # Calculate epoch statistics
    epoch_loss = running_loss / image_datasets[phase].__len__()
    epoch_acc = running_corrects.double() / image_datasets[phase].__len__()

    return epoch_loss, epoch_acc

def train(model, criterion, optimizer, num_epochs, dataloaders, device):
    start = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
    print('-' * 86)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
    
        epoch_start = time.time()
    
        # Training phase
        train_loss, train_acc = run_epoch(epoch, model, optimizer, dataloaders, device, 'Train')
        
        # Validation phase
        val_loss, val_acc = run_epoch(epoch, model, optimizer, dataloaders, device, 'Validation')
        
        epoch_time = time.time() - epoch_start
           
        # Print statistics after the validation phase
        print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                      .format(epoch + 1, train_loss, train_acc, val_loss, val_acc, epoch_time // 60, epoch_time % 60))

        # Copy and save the model's weights if it has the best accuracy thus far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    total_time = time.time() - start
    
    print('-' * 74)
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))

    # load best model weights and return them
    model.load_state_dict(best_model_wts)
    return model

#if statement is required for Windows to prevent endless loop of subprocesses, not necessary for Linux
if __name__ == '__main__': 
    model = train(model, criterion, optimizer, epochs, dataloaders, device)
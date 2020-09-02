from pathlib import Path
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
from torchvision import transforms
# from torchsummary import summary

import time
import glob
import os

DATA_DIR = Path('data')
input_size = (28, 28)
batch_size = 64
num_workers = 4
epochs = 3
learning_rate = 0.01
momentum = 0.9

# if __name__ == '__main__':   
data_transforms = {
    'Train': transforms.Compose([
                        #transforms.ToPILImage(mode=None),
                        transforms.Grayscale(num_output_channels=1),
                        # transforms.Resize(input_size),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # for RGB
                        transforms.Normalize(mean=[0.1307],std=[0.3081])                         #for grayscale

    ]),
    'Validation': transforms.Compose([
                        #transforms.ToPILImage(mode=None),
                        transforms.Grayscale(num_output_channels=1),
                        # transforms.Resize(input_size),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        transforms.Normalize(mean=[0.1307],std=[0.3081])

    ]),
    'Test': transforms.Compose([transforms.Resize(input_size),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                            transforms.Normalize(mean=[0.1307],std=[0.3081])
    ])
}

image_datasets = {x: ImageFolder(os.path.join(DATA_DIR, x),data_transforms[x])
    for x in ['Train', 'Validation']}


# dataset class to load images with no labels, for our testing set
class ImageLoader(Dataset):
    def __init__(self, root, transform=None): 
        # get image file paths
        # self.images = sorted
        self.images = sorted(glob.glob(os.path.join(root, '*')), key=self.glob_format)
        self.transform = transform
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])    #.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            return img
        else:
            return transforms.ToTensor(img)
        
    @staticmethod
    def glob_format(key):
        #key = os.path.basename(key)
        key = ''.join(c for c in key if c.isdigit())
        # key = key.split("/")[-1].split(".")[0] 
        # print(int(key))
        return "{:04d}".format(int(key))

image_datasets['Test'] = ImageLoader(str(DATA_DIR / 'Test'), transform=data_transforms['Test'])

dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers = num_workers)
                for x in ['Train', 'Validation']}

test_loader = DataLoader(dataset = image_datasets['Test'], batch_size=1, shuffle=False)



classes = image_datasets['Train'].classes
print(classes)

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}

print('Train Length: {} | Valid Length: {} | Test Length: {} '.format(dataset_sizes['Train'], dataset_sizes['Validation'], dataset_sizes['Test']))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# device = torch.device('cpu')

def get_padding(input_dim, output_dim, kernel_size, stride):
    #Calculates padding necessary to create a certain output size,
    #given a input size, kernel size and stride
    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2
    if padding < 0:
        return 0
    else:
        return padding

#------------------------------------------------------------------------
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#             #nn.Sequential() is simply a container that groups layers into one object
#             #Pass layers into it separated by commas
#         self.block1 = nn.Sequential(
#             # The first convolutional layer. Think about how many channels the input starts off with
#             # Let's have this first layer extract 32 features
#             nn.Conv2d(3, 32, 3, 1, 1),
            
#             # Don't forget to apply a non-linearity
#             nn.ReLU())
#         self.block2 =  nn.Sequential(
#             # The second convolutional layer. How many channels does it receive, given the number of features extracted by the first layer?
#             # Have this layer extract 64 features
#             nn.Conv2d(32, 64, 3, 1, 1),
        
#             # Non linearity
#             nn.ReLU(),        
#             # Lets introduce a Batch Normalization layer
#             #nn.BatchNorm2d(64),
#             # Downsample the input with Max Pooling
#             nn.MaxPool2d(2, 2, 0)
#         )
#             #Mimic the second block here, except have this block extract 128 features
#         self.block3 =  nn.Sequential(
#             nn.Conv2d(64, 128, 5, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2, 2, 0)
#         )
#             # Applying a global pooling layer
#             # Turns the 128 channel rank 4 tensor into a rank 2 tensor of size 32 x 128 (32 128-length arrays, one for each of the inputs in a batch)
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(128, 512)
#         self.drop_out = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, len(classes))
        
    
#         #Feed the input through each of the layers we defined 
#     def forward(self, x):
#         # Input size changes from (32 x 3 x 224 x 224) to (32 x 32 x 224 x 224)
#         x = self.block1(x)
#         # Size changes from (32 x 32 x 224 x 224) to (32 x 64 x 112 x 112) after max pooling
#         x = self.block2(x)
#         # Size changes from (32 x 64 x 112 x 112) to (32 x 128 x 56 x 56) after max pooling
#         x = self.block3(x)
#         # Reshapes the input from (32 x 128 x 56 x 56) to (32 x 128)
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)
#         # Fully connected layer, size changes from (32 x 128) to (32 x 512)
#         x = self.fc1(x)
#         x = self.drop_out(x)
#         # Size change from (32 x 512) to (32 x 133) to create prediction arrays for each of the images in the batch
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


#--------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#HYPERPARAMETERS WERE HERE
if __name__ == '__main__':
    model = Net()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model.to(device)


def run_epoch(epoch, model, criterion, optimizer, dataloaders, device, phase):
    running_loss = 0.0
    running_corrects = 0
    if phase == 'Train':
        model.train()
    else:
        model.eval()
    

    #Looping through batches
    for i, (inputs, labels) in enumerate(dataloaders[phase]):
    
        # ensures we're doing this calculation on our GPU if possible
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero parameter gradients
        optimizer.zero_grad()
    
        # Calculate gradients only if we're in the training phase
        with torch.set_grad_enabled(phase == 'Train'):
    
            # This calls the forward() function on a batch of inputs
            outputs = model(inputs)

            # Calculate the loss of the batch
            # loss = F.nll_loss(outputs, labels)
            loss = criterion(outputs, labels)

            # Gets the predictions of the inputs (highest value in the array)
            _, preds = torch.max(outputs, 1)

            # Adjust weights through backpropagation if we're in training phase
            if phase == 'Train':
                loss.backward()
                # print('conv1 weight: ',model.conv1.weight.grad) 
                # print('conv1 bias: ',model.conv1.bias.grad)
                # print('conv2 weight: ',model.conv2.weight.grad) 
                # print('conv2 bias: ' ,model.conv2.bias.grad)
                optimizer.step()

                if i % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, i * len(inputs), len(dataloaders['Train'].dataset),
                        100. * i / len(dataloaders['Train']), loss.item()))

        # Document statistics for the batch
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    # Calculate epoch statistics
    epoch_loss = running_loss / image_datasets[phase].__len__()
    epoch_acc = running_corrects.double() / image_datasets[phase].__len__()

    return epoch_loss, epoch_acc


def train(model, criterion, optimizer, num_epochs, dataloaders, device):
    start = time.time()
    # summary(model, (1,28, 28))

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
    print('-' * 86)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        
        epoch_start = time.time()
    
        # Training phase
        train_loss, train_acc = run_epoch(epoch, model, criterion, optimizer, dataloaders, device, 'Train')
        
        # Validation phase
        val_loss, val_acc = run_epoch(epoch, model, criterion, optimizer, dataloaders, device, 'Validation')
        
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

def test_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    #fig = plt.figure(num_images, (10, 10))
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloaders['Validation']):
        #for i, (images in enumerate(dataloaders['Validation']):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # pylint: disable=E1101
            _, preds = torch.max(outputs, 1)
            # pylint: enable=E1101
            
            for j in range(images.size()[0]):
                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                #ax.set_title('Actual: ??? \n Prediction: {}'.format(classes[preds[j]]))
                ax.set_title('Actual: {} \n Prediction: {}'.format(classes[labels[j]], classes[preds[j]]))
                print('Actual: {} , Prediction: {} '.format(classes[labels[j]], classes[preds[j]]))
            
                
                #plt.title('Actual: ??? \n Prediction: {}'.format(classes[preds[j]]))
                #print('Actual: ??? , Prediction: {} '.format(classes[preds[j]]))
                image = images.cpu().data[j].numpy().transpose((1, 2, 0))

                mean = np.array([0.5])
                std = np.array([0.5])
                
                # mean = np.array([0.5, 0.5, 0.5])
                # std = np.array([0.5, 0.5, 0.5])

                image = std * image + mean
                image = np.clip(image, 0, 1)
                #plt.imshow(image, aspect='equal')
                #plt.imshow(image)
                #plt.axis('on')
                #plt.show()
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    print('ground truth:  ',labels)
                    print('predictions:   ',preds)
                    return
            
        model.train(mode=was_training)  
                

def load_checkpoint(filepath):
    # pylint: disable=E1101
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    epoch = checkpoint['epoch']
    model.to(device)
    return model, optimizer, criterion, epoch


# if __name__ == '__main__':


if __name__ == '__main__':  
    model = train(model, criterion, optimizer, epochs, dataloaders, device)
    torch.save({
            'model' : Net(),
            'epoch' : epochs,
            'model_state_dict': model.state_dict(),
            'optimizer' : optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion' : criterion,
            'device' : device
            }, 'base_model5.tar')
    print('model saved')

    # model, optimizer, criterion, epoch = load_checkpoint('base_model5.tar')
    # test_model(model, 2)
    
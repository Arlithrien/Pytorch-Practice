import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision
import torchvision.utils
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import os
import glob
from pathlib import Path
DATA_DIR = Path('data')
num_workers =4
batch_size = 16
input_size = (128,128)


data_transforms = {
    'Train': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
    ]),
    'Validation': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
    ]),
    'Test': transforms.Compose([transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
    ])
}

image_datasets = {x: ImageFolder(os.path.join(DATA_DIR, x), data_transforms)
                for x in ['Train', 'Validation']}

#loads images w/o labels for test set              
class ImageLoader(Dataset):
    def __init__(self, root, transform=None):
        # get image file paths
        self.images = sorted(glob.glob(os.path.join(root, "*")), key=self.glob_format)
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
        key = os.path.basename(key)
        key = key.split("/")[-1].split(".")[0]
        #key = ''.join(c for c in key if c.isdigit())     
        return "{:04d}".format(int(key))

#image_datasets = data_transforms(transforms.ToPILImage()(image_datasets))
image_datasets['Test'] = ImageLoader(str(DATA_DIR / "Test"), transform=data_transforms['Test'])

dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers = num_workers)#, pin_memory=True)
            for x in ['Train', 'Validation']}

test_loader = DataLoader(dataset = image_datasets['Test'], batch_size = batch_size, shuffle=False)

classes = image_datasets['Train'].classes

#trainset = (root='./data/Train', train=True,
#    download=True, transform = transform)
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)

#valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)

#test_loader = torch.utils.data.DataLoader(train_data, sampler = test_sampler, batch_size=batch_size,num_workers=1)
'''
classes = ('0', '1', '2', '3', '4', 
            '5', '6', '7', '8', '9')
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*29*29, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x
    
net = Net()


# Loss function and optimizer
#Classification cross-entropy loss and SGD with momentum

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

#next, we loop over our data iterator, and feed the inputs to the network and optimize
net.eval()
phase = 'Train'
# iterate over test data
if __name__ == '__main__':
    for i, (inputs, target) in enumerate(dataloaders['Train']):
        # move tensors to GPU if CUDA is available
        inputs = inputs.to(device)
        labels = labels.to(device)
       $ if train_on_gpu:
         $   inputs, target = inputs.cuda(), target.cuda()
        # forward pass
        output = net(inputs)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*inputs.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    

        loss.backward()
        optimizer.step()
        # compare predictions to true label
        correct_tensor = pred.eq(target.inputs.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):       
            label = target.inputs[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1


    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    '''
    for epoch in range(2): # loop over the dataset 2 times
        running_loss =0.0
        for i, data in enumerate(trainloader, 0):
            #get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()
            
            #print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print everyu 2000 mini batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i+1, running_loss / 2000))
                running_loss = 0.0
    '''
    print('Finished Training')



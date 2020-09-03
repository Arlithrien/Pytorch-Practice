import torch
# from allinone import *

def save(Net, epochs,optimizer, model, criterion, device):
  torch.save({
              'model' : Net(),
              'epoch' : epochs,
              'model_state_dict': model.state_dict(),
              'optimizer' : optimizer,
              'optimizer_state_dict': optimizer.state_dict(),
              'criterion' : criterion,
              'device' : device
              }, 'base_model6.tar')
  print('model saved')
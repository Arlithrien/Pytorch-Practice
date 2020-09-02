import torch
from allinone import *
# from allinone import Net, epochs, model, optimizer, criterion, device

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
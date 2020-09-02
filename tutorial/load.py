import torch
from allinone import test_model
#from test import test_model

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

if __name__ == '__main__':
    model, optimizer, criterion, epoch = load_checkpoint('base_model5.tar')
    test_model(model, 4)
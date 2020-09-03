import torch
from allinone import *
# from save import *
# from allinone import Net, epochs, model, optimizer, criterion, device
# from allinone import test_model, dataloaders, device, plt
# from test import test_model

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

# def test_model(model, num_images):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     #fig = plt.figure(num_images, (10, 10))
#     print('test model')
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(dataloaders['Validation']):
#         #for i, (images in enumerate(dataloaders['Validation']):
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             # pylint: disable=E1101
#             _, preds = torch.max(outputs, 1)
#             # pylint: enable=E1101
            
#             for j in range(images.size()[0]):
#                 images_so_far += 1

#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 #ax.set_title('Actual: ??? \n Prediction: {}'.format(classes[preds[j]]))
#                 ax.set_title('Actual: {} \n Prediction: {}'.format(classes[labels[j]], classes[preds[j]]))
#                 print('Actual: {} , Prediction: {} '.format(classes[labels[j]], classes[preds[j]]))
            
                
#                 #plt.title('Actual: ??? \n Prediction: {}'.format(classes[preds[j]]))
#                 #print('Actual: ??? , Prediction: {} '.format(classes[preds[j]]))
#                 image = images.cpu().data[j].numpy().transpose((1, 2, 0))

#                 mean = np.array([0.5])
#                 std = np.array([0.5])
                
#                 # mean = np.array([0.5, 0.5, 0.5])
#                 # std = np.array([0.5, 0.5, 0.5])

#                 image = std * image + mean
#                 image = np.clip(image, 0, 1)
#                 #plt.imshow(image, aspect='equal')
#                 #plt.imshow(image)
#                 #plt.axis('on')
#                 #plt.show()
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     print('ground truth:  ',labels)
#                     print('predictions:   ',preds)
#                     return
            
#         model.train(mode=was_training)  


if __name__ == '__main__':
    model, optimizer, criterion, epoch = load_checkpoint('base_model6.tar')
    test_model(model, 4)
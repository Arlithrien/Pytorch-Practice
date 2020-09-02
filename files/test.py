import torch 
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
from allinone import device, test_loader, classes, dataloaders, train


ImageFile.LOAD_TRUNCATED_IMAGES = True


def test_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    #fig = plt.figure(num_images, (10, 10))

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloaders['Validation']):
            #for images in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # pylint: disable=E1101
            _, preds = torch.max(outputs, 1)
            # pylint: enable=E1101
            print(images.size()[3])
            for j in range(images.size()[0]):
                images_so_far += 1

                #ax = plt.subplot(num_images//1, 1, images_so_far)
                #ax.axis('off')
                #ax.set_title('Actual: {} \n Prediction: {}'.format(gauge_ranges[labels[j]], gauge_ranges[preds[j]]))
                #ax.set_title('Actual: {} \n Prediction: {}'.format(gauge_ranges[labels[j]], gauge_ranges[preds[j]]))
                #print('Actual: {} , Prediction: {} '.format(gauge_ranges[labels[j]], gauge_ranges[preds[j]]))
                plt.title('Actual: {} \n Prediction: {}'.format(classes[labels[j]], classes[preds[j]]))
                print('Actual: {} , Prediction: {} '.format(classes[labels[j]], classes[preds[j]]))
                image = images.cpu().data[j].numpy().transpose((1, 2, 0))

                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                plt.imshow(image, aspect='equal')
                plt.axis('on')
                plt.show()
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#model = train(model, criterion, optimizer, epochs, dataloaders, device)
#model, optimizer, criterion, epoch = load_checkpoint('base_model3.tar')
#test_model(model, 6)
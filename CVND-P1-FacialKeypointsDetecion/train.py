import torch
from models import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import os

def train_net(n_epochs):
    # prepare the net for training
    net.train()
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        print("epoch " + str(epoch + 1))
        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            print("batch")
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')



package_dir = os.path.dirname(os.path.abspath(__file__))

net = Net()
# net.load_state_dict(torch.load(package_dir + '/saved_models/model3.pt'))
print(net)


data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

##########LOAD TRAIN DATASET
transformed_dataset = FacialKeypointsDataset(csv_file= package_dir + '/data/training_frames_keypoints.csv',
                                             root_dir= package_dir + '/data/training/',
                                             transform=data_transform)

batch_size = 30
train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

##########LOAD TEST DATASET
test_dataset = FacialKeypointsDataset(csv_file=package_dir+'/data/test_frames_keypoints.csv',
                                             root_dir=package_dir+'/data/test/',
                                             transform=data_transform)

test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

print('Number of images: ', len(transformed_dataset))



criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(params=net.parameters(),lr=0.01)


n_epochs = 3
train_net(n_epochs)

#Save Model
model_dir = package_dir +'/saved_models/'
model_name = 'model4.pt'

torch.save(net.state_dict(), model_dir+model_name)
net.eval()
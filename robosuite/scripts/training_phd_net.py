import pickle
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from robosuite.mcts.resnet import ResNet, BasicBlock, Bottleneck
from robosuite.mcts.inception import Inceptionv4

class TransitionDataset(Dataset):
    def __init__(self, data_root_prefix='robosuite/data/img_data', num_data_per_file=128, total_file=234):
        self.total_file = total_file
        self.num_data_per_file = num_data_per_file
        self.total_num = num_data_per_file*total_file
        self.data_root_prefix = data_root_prefix

        print("Dataset statistics")
        print("Number of data : {}".format(self.total_num))

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        data_idx = idx%self.num_data_per_file
        data_file_idx = int(idx/self.num_data_per_file)

        with open(self.data_root_prefix+str(data_file_idx)+'.pkl', 'rb') as f:
            data = pickle.load(f)
            depth1_list = data['depth1_list']
            depth2_list = data['depth2_list']
            mask1_list = data['mask1_list']
            mask2_list = data['mask2_list']
            label_list = data['label']

            depth1 = depth1_list[data_idx]
            depth2 = depth2_list[data_idx]
            mask1 = mask1_list[data_idx]
            mask2 = mask2_list[data_idx]
            label = label_list[data_idx]

        return (np.asarray(depth1),
                np.asarray(depth2),
                np.asarray([mask1], np.float32),
                np.asarray([mask2], np.float32),
                np.asarray(label, np.long))

# class TransitionDataset(Dataset):
#     def __init__(self, data_root='robosuite/data/img_data.pkl'):
#         self.samples = []
#         self.succ_rate = 0
#         self.total_num = 0
#
#         with open(data_root, 'rb') as f:
#             data = pickle.load(f)
#             depth1_list = data['depth1_list']
#             depth2_list = data['depth2_list']
#             mask1_list = data['mask1_list']
#             mask2_list = data['mask2_list']
#             label_list = data['label']
#
#         print("Dataset statistics")
#         print("Number of data : {}".format(len(label_list)))
#         print("True labeled data: {}".format(np.mean(label_list)))
#
#         self.total_num = len(label_list)
#         for depth1, depth2, mask1, mask2, label in zip(depth1_list, depth2_list, mask1_list, mask2_list, label_list):
#             self.samples.append((np.asarray(depth1),
#                                  np.asarray(depth2),
#                                  np.asarray([mask1], np.float32),
#                                  np.asarray([mask2], np.float32),
#                                  np.asarray(label, np.long)))
#             self.succ_rate += float(label) / self.total_num
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         return self.samples[idx]


if __name__ == "__main__":
    # model_name_list = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNext50', 'Wide_ResNet50', 'InceptionV4']
    model_name_list = ['ResNet34', 'ResNet50', 'ResNext50', 'Wide_ResNet50']

    transition_dataset = TransitionDataset()
    total_dataset = transition_dataset.total_num
    training_len = int(total_dataset*0.7)
    val_len = total_dataset - training_len
    train_set, val_set = torch.utils.data.random_split(transition_dataset, [training_len, val_len])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    training_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=val_len)

    for model_name in model_name_list:
        if model_name in 'ResNet18':
            model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device) # ResNet18

            # Hyper-parameters
            early_stop = None
            num_epochs = 30
            learning_rate = 0.001
        elif model_name in 'ResNet34':
            model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device) # ResNet34

            # Hyper-parameters
            early_stop = 45
            num_epochs = 60
            learning_rate = 0.001
        elif model_name in 'ResNet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device) # ResNet50

            # Hyper-parameters
            early_stop = 45
            num_epochs = 60
            learning_rate = 0.001
        elif model_name in 'ResNext50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4).to(device) # ResNext50

            # Hyper-parameters
            early_stop = 45
            num_epochs = 60
            learning_rate = 0.001
        elif model_name in 'Wide_ResNet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64*2).to(device) # Wide ResNet50

            # Hyper-parameters
            early_stop = 45
            num_epochs = 60
            learning_rate = 0.001
        elif model_name in 'InceptionV4':
            model = Inceptionv4().to(device)

            # Hyper-parameters
            early_stop = None
            num_epochs = 30
            learning_rate = 0.0001

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # For updating learning rate
        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Train the model
        total_step = len(training_loader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            for i, batch in enumerate(training_loader):
                images = torch.cat(batch[:2], dim=1).to(device)
                labels = batch[4].to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                training_acc = 100. * correct / total

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    print("Model Name: {}, Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, Accuracy: {} %"
                          .format(model_name, epoch + 1, num_epochs, i + 1, total_step, loss.item(), training_acc))

            # Decay learning rate
            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

            # Validate the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, batch in enumerate(training_loader):
                    images = torch.cat(batch[:2], dim=1).to(device)
                    labels = batch[4].to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                test_acc = 100. * correct / total
                print('Accuracy of the model on the test images: {} %'.format(test_acc))

            if early_stop is not None and (epoch + 1) == early_stop:
                torch.save(model.state_dict(), 'robosuite/networks/' + model_name + '_depth_img_early_stop' + '.pt')
                with open('robosuite/networks/' + model_name + '_depth_img_early_stop' + '_results.pkl', 'wb') as f:
                    pickle.dump({'training_acc': training_acc, 'test_acc': test_acc}, f, pickle.HIGHEST_PROTOCOL)

        torch.save(model.state_dict(), 'robosuite/networks/'+model_name+'_depth_img'+'.pt')
        with open('robosuite/networks/'+model_name+'_depth_img'+'_results.pkl', 'wb') as f:
            pickle.dump({'training_acc': training_acc,'test_acc': test_acc}, f, pickle.HIGHEST_PROTOCOL)

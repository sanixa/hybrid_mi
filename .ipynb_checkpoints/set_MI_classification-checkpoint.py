# +
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random, copy
import re

import model, log

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST

from datasets.celeba import CelebA

from sklearn import metrics


# -

# definition of inference model 
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=1):
        super(Net, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Tanh())
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_dim, output_dim))


        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        logits = self.layers(x)
        return torch.sigmoid(logits)


def test_classifier(test_data, test_y, net, normalize=False):

    if(normalize):
        test_data = test_data / test_data.max(axis=0) # normalize each cloumn
    test_data, test_y = torch.tensor(test_data, dtype=torch.float), torch.tensor(test_y, dtype=torch.float)
    test_data, test_y = test_data.cuda(), test_y.cuda()
    probs = net(test_data)
    
    ###AUCROC
    auc = metrics.roc_auc_score(test_y.cpu().detach().numpy(), probs.cpu().detach().numpy())
    
    probs[probs>0.5] = 1
    probs[probs<0.5] = 0
    num_correct = torch.sum(probs==test_y)
    accuracy = 1.0 * num_correct.item() / test_data.shape[0]
    

    half = int(test_data.shape[0]/2)
    trn_accuracy = torch.sum(probs[0:half]==test_y[0:half]).item() *1.0 / half
    test_accuracy = torch.sum(probs[half:]==test_y[half:]).item() *1.0 / half
    return accuracy, trn_accuracy, test_accuracy, auc


# train the inference model
def train_classifier(trn_data, trn_y, test_data, test_y, hidden_dim=100, layers=3, T=1000, batchsize=200, lr=0.5, momentum=0., normalize=False):

    if(normalize):
        trn_data = trn_data / trn_data.max(axis=0) # normalize each cloumn
    trn_data, trn_y = torch.tensor(trn_data, dtype=torch.float), torch.tensor(trn_y, dtype=torch.float)
    trn_data, trn_y = trn_data.cuda(), trn_y.cuda()
    
    loss_func = F.binary_cross_entropy
    net = Net(trn_data.shape[1], hidden_dim, 1, layers).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=0.)
    best_auc = -1
    eval_freq=5

    num_trn_samples = trn_data.shape[0]
    for t in range(T):
        idx = np.random.choice(num_trn_samples, batchsize, replace=False)
        probs = net(trn_data[idx])
        loss = loss_func(probs, trn_y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(t%eval_freq==0):
            cur_acc, cur_trn_acc, cur_test_acc, cur_auc = test_classifier(test_data, test_y, net, normalize)
            if(cur_auc > best_auc):
                best_auc = cur_auc
                best_acc = cur_acc
                best_net = copy.deepcopy(net)
                best_trn_acc = cur_trn_acc
                best_test_acc = cur_test_acc
                print('curr best auc/acc:{}, {}'.format(best_auc, best_acc))
    return best_net, best_auc, best_acc, best_trn_acc, best_test_acc#best_net


def get_moments(arr, order=2):
    power_arr = np.power(arr, order)
    mean_arr = np.mean(power_arr, axis=1)
    moment = np.power(mean_arr, 1./order)
    return moment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--resolution', type=int, default=64,
                        help='target image resolution')
    parser.add_argument('--seed', '-seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--data_num', '-dnum', type=int, default=10000,
                        help='the number of query images to be considered')
    parser.add_argument('--batch_size', '-bs', type=int, default=100,
                        help='batch size (should not be too large for better optimization performance)')
    parser.add_argument('--cnn-path-dir', '-cdir', type=str, default=None,
                        help='file path of classifier')
    args = parser.parse_args()
    
    ### set up classifier
    if not os.path.isdir(args.cnn_path_dir):
        sys.exit("dir does not exist")
    path_list = os.listdir(args.cnn_path_dir)
    model_list = []
    for file in path_list:
        #import ipdb;ipdb.set_trace()
        if re.match(r'^.*.pth', file):
            netC = model.cnn_celeba().cuda()
            netC.load_state_dict(torch.load(os.path.join(args.cnn_path_dir, file)))
            netC.eval()
            model_list.append(netC)
            
    
    transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = CelebA(root=os.path.join('/work/u5366584/exp/datasets/celeba'), split='train', #/work/u5366584/exp/datasets/celeba \\ ../intern/GS-WGAN-custom/exp/datasets/celeba
        transform=transform_train, download=False)
    testset = CelebA(root=os.path.join('/work/u5366584/exp/datasets/celeba'), split='test', #/work/u5366584/exp/datasets/celeba \\ ../intern/GS-WGAN-custom/exp/datasets/celeba
        transform=transform_train, download=False)
    
    indices = np.loadtxt('index_20k.txt', dtype=np.int_)
    trainset = torch.utils.data.Subset(trainset, indices)
    print(len(trainset))
    workers = 2
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss(reduction='none')

    Gender = 20
    iters = int(args.data_num / args.batch_size)
    
    train_attack_x = []
    train_attack_y = []
    with torch.no_grad():
        for i, (train_data, train_target) in enumerate(train_loader):
            train_data, train_target = train_data.float().cuda(), train_target[:, Gender].long().cuda()
            temp_x = []
            for j in range(len(model_list)):
                output = model_list[j](train_data)
                train_loss = criterion(output, train_target)
                temp_x.append(train_loss.cpu().numpy().reshape(-1,1)) ##loss
            
            train_attack_x.append(np.hstack(temp_x))
            if i+1 == iters:
                break
    train_attack_x = np.vstack(train_attack_x)
    train_attack_y = np.ones(train_attack_x.shape[0]).reshape(-1,1) ##membership label 1
        
    test_attack_x = []
    test_attack_y = []
    with torch.no_grad():
        for i, (test_data, test_target) in enumerate(test_loader):
            test_data, test_target = test_data.float().cuda(), test_target[:, Gender].long().cuda()
            temp_x = []
            for j in range(len(model_list)):
                output = model_list[j](test_data)
                test_loss = criterion(output, test_target)
                temp_x.append(test_loss.cpu().numpy().reshape(-1,1)) ##loss
            
            test_attack_x.append(np.hstack(temp_x))
            if i+1 == iters:
                break
    test_attack_x = np.vstack(test_attack_x)
    test_attack_y = np.zeros(test_attack_x.shape[0]).reshape(-1,1) ##membership label 0
                
    #### follow  https://github.com/dayu11/MI_with_DA/blob/main/mi_attack.py#L89  ####Algorithm 4 from paper
    
    print('inference accuracy of moments feature')
    trn_features = []
    test_features = []
    
    for order in range(1, 21):  # we use moments with orders 1~20
        trn_moments = get_moments(train_attack_x, order).reshape([-1, 1])
        test_moments = get_moments(test_attack_x, order).reshape([-1, 1])
        trn_features.append(trn_moments)
        test_features.append(test_moments)
    trn_features = np.concatenate(trn_features, axis=1) # size:10000x20
    test_features = np.concatenate(test_features, axis=1) # size:10000x20
    
    # we use 1000 exampls to train inference model
    train_data = np.concatenate([trn_features[:1000], test_features[:1000]], axis=0)
    train_y = np.concatenate([train_attack_y[:1000], test_attack_y[:1000]], axis=0)
    # we use 9000 examples to evaluate inference accuracy
    test_data = np.concatenate([trn_features[1000:], test_features[1000:]], axis=0)
    test_y = np.concatenate([train_attack_y[1000:], test_attack_y[1000:]], axis=0)
    
    net, best_auc, best_acc, best_trn_acc, best_test_acc = train_classifier(train_data, train_y, test_data, test_y, hidden_dim=20, layers=1)
    print(best_acc, 'accuracy on training/test set: ', best_trn_acc, best_test_acc)
    print('AUCORC: ', best_auc)

if __name__ == '__main__':
    main()

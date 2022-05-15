#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, fmnist_iid, svhn_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img



def DeltaWeight(w1, w2):
    diff = 0
    norm1 = 0
    norm2 = 0
    all_dot = 0

    for k in w1.keys():
        param1 = w1[k]
        param2 = w2[k]

        curr_diff = torch.norm(param1 - param2, p='fro')
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        all_dot += torch.sum(param1 * param2)
        diff += curr_diff * curr_diff
    return all_dot / torch.sqrt(norm1 * norm2)


def test():
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)           
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist', train=False, download=True, transform=trans_fmnist)
        if args.iid:
            dict_users = fmnist_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in FMNIST')
    elif args.dataset == 'svhn':
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.SVHN('../data/svhn',  split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN('../data/svhn', split='test', download=True, transform=trans_svhn)
        #dataset_extra = datasets.SVHN('../data/svhn', split='extra', transform=trans_svhn,
         #                        target_transform=None, download=True)
        if args.iid:
            dict_users = svhn_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in SVHN')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    w_init = copy.deepcopy(net_glob.state_dict())

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    delete_round = args.delete_round
    delete_data_ratio = args.delete_data_ratio
    delete_client_ratio = args.delete_client_ratio

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        net_glob.train()

        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)
        test()
        '''
         在指定的 epoch 进行截断
         -  删除比例clients的比例数据集
        '''
        if iter == delete_round:
            break

    # net_glob: server model
    # w_locals: 每个client的model list
    old_w_glob = copy.deepcopy(w_glob)
    old_w_locals = copy.deepcopy(w_locals)

    print('-'*50)
    # RETRAIN
    print('RETRAIN')

    # 重新定义参数
    net_glob.load_state_dict(w_init)
    w_glob = net_glob.state_dict()

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    threshold_w = args.threshold_w
    w_loss = 1000
    round = 0

    # 删除比例数据
    delete_number = max(int(args.delete_client_ratio * args.num_users * args.frac), 1)
    print(delete_number)
    delete_users = np.random.choice(range(args.num_users), delete_number, replace=False)

    print('before:',len(dict_users[delete_users[0]]))
    user_sample_len = len(dict_users[0])
    delete_sample_number = max(int(args.delete_data_ratio * user_sample_len), 1)
    print(delete_sample_number)
    for delete_user in delete_users:
        delete_index = np.random.choice(range(user_sample_len), delete_sample_number, replace=False)
        dict_users[delete_user] = set(np.delete(list(dict_users[delete_user]), delete_index))
    print('after:',len(dict_users[delete_users[0]]))

    time_s = 0
    # while w_loss > threshold_w:
    new_epoch = args.new_epoch
    for i in range(new_epoch):
        net_glob.train()

        start_time = time.perf_counter()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        round += 1
        print('Round {:3d}, Average loss {:.3f}'.format(round, loss_avg))
        end_time = time.perf_counter()

        time_s += end_time - start_time
        test()
        #w_loss = DeltaWeight(old_w_glob, w_glob)
        #print('Delta Weight:{:.3f}'.format(w_loss))


    print('Retrain Time : {:.3f}s'.format(time_s))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

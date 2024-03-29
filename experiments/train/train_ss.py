'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import os
import time

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir,
                                             "scale-equivariant-steerable")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from models.mnist_ss import ss_classification_224
from dataloaders import make_aar_loaders, make_imagenet_loaders
from utils.train_utils import train_xent, test_acc
from utils.model_utils import get_num_parameters
from utils.misc import dump_list_element_1line


def train_ss(
        dataloader_name,
        batch_size=64,
        scaling_factor=1,
        lr=0.01,
        lr_steps=[20, 40],
        lr_gamma=0.1,
        save_model_path=None,
        epochs=10
):
    #########################################
    # model configuration
    #########################################
    # model_names = sorted(name for name in models.__dict__
    #                      if name.islower() and not name.startswith("__")
    #                      and callable(models.__dict__[name]))

    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--epochs', type=int, default=60)

    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--nesterov', action='store_true', default=False)
    # parser.add_argument('--decay', type=float, default=1e-4)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--lr_steps', type=int, nargs='+', default=[20, 40])
    # parser.add_argument('--lr_gamma', type=float, default=0.1)

    # parser.add_argument('--model', type=str, choices=model_names, required=True)
    # parser.add_argument('--extra_scaling', type=float, default=1.0,
    #                    required=False, help='add scaling data augmentation')
    # parser.add_argument('--cuda', action='store_true', default=False)
    # parser.add_argument('--save_model_path', type=str, default='')
    # parser.add_argument('--tag', type=str, default='', help='just a tag')
    # parser.add_argument('--data_dir', type=str)

    # args = parser.parse_args()

    # print("Args:")
    # for k, v in vars(args).items():
    #     print("  {}={}".format(k, v))

    # print(flush=True)
    # assert len(args.save_model_path)

    #########################################
    # Data
    #########################################
    if dataloader_name == "imagenet_classify":
        train_loader, val_loader, test_loader = make_imagenet_loaders(batch_size, scaling_factor, num_output_channels=1)
    elif dataloader_name == "aar":
        train_loader, val_loader, test_loader = make_aar_loaders(batch_size, scaling_factor, num_output_channels=1)

    # print('Train:')
    # print(loaders.loader_repr(train_loader))
    # print('\nVal:')
    # print(loaders.loader_repr(val_loader))
    # print('\nTest:')
    # print(loaders.loader_repr(test_loader))

    #########################################
    # Model
    #########################################
    # model = models.__dict__[args.model]
    # model = model(**vars(args))

    model = ss_classification_224(num_classes=6)
    print('\nModel:')
    print(model)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device: {}'.format(device))

    if use_cuda:
        cudnn.enabled = True
        cudnn.benchmark = True
        print('CUDNN is enabled. CUDNN benchmark is enabled')
        model.cuda()

    print('num_params:', get_num_parameters(model))
    print(flush=True)

    #########################################
    # optimizer
    #########################################
    parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=lr)

    # if args.optim == 'adam':
    #     optimizer = optim.Adam(parameters, lr=args.lr)
    # if args.optim == 'sgd':
    #     optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
    #                           weight_decay=args.decay, nesterov=args.nesterov)

    print(optimizer)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)

    #########################################
    # training
    #########################################
    print('\nTraining\n' + '-' * 30)

    if save_model_path:
        if not os.path.isdir(os.path.dirname(save_model_path)):
            os.makedirs(os.path.dirname(save_model_path))

    start_time = time.time()
    best_acc = 0.0
    train_acc_list = list()
    val_acc_list = list()

    for epoch in range(epochs):
        train_acc = train_xent(model, optimizer, train_loader, device, batch_size)
        val_acc = test_acc(model, val_loader, device)
        print('Epoch {:3d}/{:3d}| Acc@1: {:3.1f}%'.format(
            epoch + 1, epochs, 100 * val_acc), flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_model_path)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        lr_scheduler.step()

    print('-' * 30)
    print('Training is finished')
    print('Best Acc@1: {:3.1f}%'.format(best_acc * 100), flush=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_epoch = elapsed_time / epochs

    print('\nTesting\n' + '-' * 30)
    model.load_state_dict(torch.load(save_model_path))
    final_acc = test_acc(model, test_loader, device)
    print('Test Acc:', final_acc)

    #########################################
    # save results
    #########################################
    # results = vars(args)
    # results.update({
    #     'dataset': 'scale_mnist',
    #     'elapsed_time': int(elapsed_time),
    #     'time_per_epoch': int(time_per_epoch),
    #     'num_parameters': int(get_num_parameters(model)),
    #     'acc': final_acc,
    # })

    # with open('results.yml', 'a') as f:
    #     f.write(dump_list_element_1line(results))

    results = {
        'dataset': dataloader_name,
        'elapsed_time': int(elapsed_time),
        'time_per_epoch': int(time_per_epoch),
        'num_parameters': int(get_num_parameters(model)),
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'final_acc': final_acc
    }

    return results


if __name__ == '__main__':
    train_ss()

import argparse

import torch
from torch import nn
from torchvision import transforms

from tl.augmentation import GaussianBlur
from loader.dataset import StanfordCarsDataset, AircraftDataset, CIFAR100Dataset, DTDDataset, dogs
from models.generate_model import build_model
from opt.train import train, validate
from utils.cosine_decay import adjust_learning_rate
from utils.early_stopping import EarlyStopping
from utils.plots import save_plots

# Construct the argument parser.
parser = argparse.ArgumentParser(description='PyTorch MocoV2 pre-training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ')
parser.add_argument(
    '-e', '--epochs', default=100, type=int,
    help='Number of epochs to train our network for'
)
parser.add_argument('--lr', default=0.01, type=float,
                    help='Learning rate for training the model'
                    )
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N')
parser.add_argument('--schedule', default=[25, 35], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--model', default=None, type=str,
                    help='pretrained model')
parser.add_argument('--runSchedule', action='store_true', default=False,
                    help='Decide if ReducePlateau schedule used')
parser.add_argument('--adjustLR', action='store_true', default=False,
                    help='adjust LR')
parser.add_argument('--isCheckpoint', action='store_true', default=False,
                    help='adjust LR')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dataset', default='', type=str,
                    help='dataset used for train/test')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--lrDecay', default=40.0, type=float,
                    help='LR decay used in adjustLR')

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose([  # transforms.Resize((400, 400)),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_tfms = transforms.Compose([  # transforms.Resize((400, 400)),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == 'stanfordCars':
        train_dataset = StanfordCarsDataset(root='dataTrain', split='train', transform=train_tfms, download=True)
        test_dataset = StanfordCarsDataset(root='dataTest', split='test', transform=test_tfms, download=True)
    elif args.dataset == 'aircraft':
        train_dataset = AircraftDataset(root='dataAircraftTrain', split='train', transform=train_tfms, download=True)
        test_dataset = AircraftDataset(root='dataAircraftTest', split='test', transform=test_tfms, download=True)
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100Dataset(root='CifarDataTrain', train=True, transform=train_tfms, download=True)
        test_dataset = CIFAR100Dataset(root='CifarDataTest', transform=test_tfms, download=True)
    elif args.dataset == 'dtd':
        train_dataset = DTDDataset(root='DTDTrain', split='train', transform=train_tfms, download=True)
        train_dataset2 = DTDDataset(root='DTDVal', split='val', transform=train_tfms, download=True)
        test_dataset = DTDDataset(root='DTDTest', split='test', transform=test_tfms, download=True)
    elif args.dataset == 'dogs':
        train_dataset = dogs(root='stanfordDogs', train=True, transform=train_tfms, download=True)
        test_dataset = dogs(root='stanfordDogs', train=False, transform=test_tfms, download=True)

    train_dataset_classes = train_dataset.classes
    print('running dataset: ', args.dataset)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    if args.dataset == 'dtd':
        trainloader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=12)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate model
    model = build_model(pretrainedPath=args.model, num_classes=len(train_dataset_classes), args=args).to(device)
    early_stopping = EarlyStopping(patience=8, verbose=True, delta=0.0001, mode='max', model=model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4,
                                                             threshold=0.005)

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_acc = 0.0
    # Start the training.
    epochs = args.epochs
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        if args.adjustLR:
            adjust_learning_rate(optimizer, epoch + 1, args)

        train_epoch_loss, train_epoch_acc = train(model, trainloader,
                                                  optimizer, criterion)
        if args.dataset == 'dtd':
            train_epoch_loss, train_epoch_acc = train(model, trainloader2,
                                                      optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, testloader,
                                                     criterion, train_dataset_classes, lrscheduler, args)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        early_stopping(valid_epoch_acc, model)

        if early_stopping.counter == 6:
            model = early_stopping.model
            print('model changed')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # Checking for best accuracy
        for param_group in optimizer.param_groups:
            newLR = param_group['lr']
        print(newLR)

        is_best = valid_epoch_acc > best_acc
        if is_best:
            best_acc = valid_epoch_acc
            print('best accuracy:', best_acc)
            torch.save(model, args.dataset + '_best_model.pth.tar')
        print('-' * 50)
    print('TRAINING COMPLETE')
    torch.save(model, args.dataset + '_final_model.pth.tar')
    save_plots(train_acc, valid_acc, train_loss, valid_loss)


if __name__ == '__main__':
    main()

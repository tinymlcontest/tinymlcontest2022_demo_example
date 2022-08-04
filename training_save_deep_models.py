import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from util import *
from models.model_1 import IEGMNet, perceptron
from models.Mobilenet_V2 import mobilenetv2
from torchsummary import summary
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def main(net):
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_TEST = args.batch_size
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    # create logger for ploting
    file_complete = creating_path("Result","logs", args.model,
                                  file_name=args.model, extension='log')
    logger_complete = create_logger("complete", file_complete)

    print("Start training")
    best_acc = 0
    net = net.to(device)
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)
        net.train()
        running_loss = 0.0
        correct = 0.0
        train_accuracy = 0.0
        total = 0.0
        i = 0
        for j, data in enumerate(trainloader):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            running_loss += loss.item()
            i += 1
        train_accuracy = correct / total
        Train_loss = running_loss / i
        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, train_accuracy, Train_loss))

        running_loss = 0.0
        test_accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        with torch.no_grad():
            for data_test in testloader:
                net.eval()
                IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
                IEGM_test = IEGM_test.float().to(device)
                labels_test = labels_test.to(device)
                outputs_test = net(IEGM_test)
                _, predicted_test = torch.max(outputs_test.data, 1)
                total += labels_test.size(0)
                correct += (predicted_test == labels_test).sum()

                loss_test = criterion(outputs_test, labels_test)
                running_loss_test += loss_test.item()
                i += 1
        test_accuracy = (correct / total).item()
        Test_loss = running_loss_test / i
        print('Test Acc: %.5f Test Loss: %.5f' % (test_accuracy, Test_loss))
        
        if best_acc < test_accuracy:
            best_acc = test_accuracy
            torch.save(net, './saved_models/' + args.model+ '.pkl')
            torch.save(net.state_dict(), './saved_models/' + args.model+ '_state_dict.pkl')

        cur_lr = optimizer.param_groups[0]['lr']
        msg = ('Epoch: [{0}]\t'
                'LR:[{1}]\t'
                'Train_acc {2}\t'
                'Train_loss {3}\t'
                'Test_acc {4}\t'
                'Test_loss {5}\t'
                )
        logger_complete.info(msg.format(epoch+1, cur_lr, train_accuracy, Train_loss, test_accuracy, Test_loss))

    closer_logger(logger_complete)
    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batch_size', type=int, help='total batch size for traindb', default=64)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--model', type=str, default='IEGM_net')
    args = argparser.parse_args()

    model_dic = {
    "IEGM_net": IEGMNet(),
    "mobilenetv2_simple": mobilenetv2(num_classes=2, width_mult=1.,last_channel = 640)
    }

    if not os.path.exists('./saved_models/'):
        os.makedirs(args.path_net)
    if torch.cuda.is_available():    
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = 'cpu'

    print("device is --------------", device)

    # Instantiating NN
    net = model_dic[args.model]
    summary(net.to(device), (1,1250,1))

    main(net)

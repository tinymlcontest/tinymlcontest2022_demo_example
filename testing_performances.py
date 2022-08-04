import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import torchvision.transforms as transforms
import os
from util import *


def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    BATCH_SIZE_TEST = 1
    SIZE = args.size
    path_data = args.path_data
    path_records = args.path_record
    path_net = args.path_net
    path_indices = args.path_indices
    stats_file = open(path_records + 'seg_stat.txt', 'w')

    # load trained network
    net = torch.load(path_net + args.model+ '.pkl', map_location = device)
    net.eval()
    net.to(device)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    with torch.no_grad():
        for data_test in testloader:
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            seg_label = deepcopy(labels_test)

            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)

            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)

            if seg_label == 0:
                segs_FP += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
                segs_TN += (predicted_test == labels_test).sum().item()
            elif seg_label == 1:
                segs_FN += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
                segs_TP += (predicted_test == labels_test).sum().item()

    # report metrics
    stats_file.write('segments: TP, FN, FP, TN\n')
    output_segs = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    stats_file.write(output_segs + '\n')

    del net


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    argparser.add_argument('--path_record', type=str, default='./records/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--model', type=str, default='IEGM_net')
    args = argparser.parse_args()

    if not os.path.exists(args.path_net):
        os.makedirs(args.path_net)
    if not os.path.exists(args.path_record):
        os.makedirs(args.path_net)

    if torch.cuda.is_available():    
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = 'cpu'
    print("device is --------------", device)

    main()

import argparse
import time
import datetime
import os
import shutil
# import sys
import numpy as np

import pandas as pd
import cv2
import math as m
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from torchvision import transforms
import matplotlib.pyplot as plt
from data_util_forsegformer import data_loader
# from datautilforsub import data_loader as sub_loader
# from dataUtilEvalstagetwo import data_loader as sub_loader
# from network.get_model import get_model
from network.CANet import CANet
from network.dv3 import DeepLabV3
from network.segformer import SegFormer
from pre import get_boundary_pos
from network.SCNet import SCNet
from loss.loss import cross_entropy_loss, miou_loss, smooth_L1_loss
from util.image_processor import rampcolor
from util.metrics import Metric_binary, Metric_tdf


def parse_args():
    parser = argparse.ArgumentParser(description='FireNet With Pytorch')

    parser.add_argument('--model', type=str, default='unet',
                        choices=['fcn32s', ' '],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['vgg16', 'resnet18'],
                        help='backbone name (default: vgg16)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--save-dir', default='./pth/',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--data-path', type=str, default='./data_test/dataset/result/',
                        help='the data path for train')

    parser.add_argument('--path-predict', type=str, default='./data/burris_train/predict_R25/',
                        help='the data path for train')

    parser.add_argument('--path-model', type=str,
                        default='./pth/GeoMAC/210915_exp120_4000x10samples_dv3_res50_l1_R2_5/segwind_10pixel_360_best_valid_model_8.25_save.pth',
                        help='the model path for train')
    #
    parser.add_argument('--path-sub', type=str,
                        default='./pth/GeoMAC/210915_exp120_4000x10samples_dv3_res50_l1_R2_5/segwind_sub_180_best_valid_model.pth',
                        help='the model path for train')
    parser.add_argument('--time-interval', type=int, default=30,
                        help='time interval, unit minute')
    #
    parser.add_argument('--time-delt', type=int, default=30,
                        help='time delt for deduce')
    #
    parser.add_argument('--cell-length', type=int, default=30,
                        help='time interval for fire spread')
    #
    parser.add_argument('--hours', type=int, default=12,
                        help='12 hours for simulation')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    device = "cuda"

    num_iter = args.hours

    model_sub = SegFormer('MiT-B4').to(device)

    model_sub.load_state_dict(torch.load(args.path_model,map_location=device)['state_dict_CANet'])
    model_sub.eval()

    sub_dataset = data_loader(args.data_path)

    paths_list = sub_dataset.paths_list

    start_time = time.time()

    accuracy_list = np.zeros((len(paths_list), num_iter))
    mious_list = np.zeros((len(paths_list), num_iter))
    f1_list = np.zeros((len(paths_list), num_iter))
    time_mte_list = np.zeros((len(paths_list), num_iter))
    ide_list = np.zeros(len(paths_list))
    ide_list_sub = np.zeros(len(paths_list))
    li_ide = []
    li_ide_N = []
    li_AC = []

    for i in range(len(paths_list)):

        inputs_sub,targets = sub_dataset.__getitem__(i)

        inputs_sub = inputs_sub[0]

        fire_perimeter = inputs_sub.data.cpu().numpy()[0,:,:]

        pere_number = get_boundary_pos(fire_perimeter).shape[0]

        inputs_sub = inputs_sub.unsqueeze(0).to(device)

        outputs_sub = model_sub(inputs_sub)

        outputs_sub = F.softmax(outputs_sub,dim=1)
        targets_td = targets[0]

        targets_td = targets_td.data.cpu().numpy()
        targets_td_ac = targets_td + 1
        targets_td_ac[targets_td_ac > 2] = 0
        targets_td_ac[targets_td_ac > 1] = 1

        np_outputs_sub = outputs_sub.data.cpu().numpy()

        np_output_sub = np_outputs_sub[0, :, :, :]


        for k in range(np_output_sub.shape[0]):
            for p in range(np_output_sub.shape[1]):
                for j in range(np_output_sub.shape[2]):
                    if (np_output_sub[k, p, j] == np.max(np_output_sub[:, p, j])):
                        np_output_sub[k, p, j] = 1
        np_output_1 = np_output_sub[0,:,:]
        np_output_1[np_output_1<1] = 0
        np_output_2 = np_output_sub[1,:,:]
        np_output_2[np_output_2<1] = 0
        np_output_AC = np_output_1+np_output_2

        np_test = targets_td_ac + np_output_AC
        np_result = (np_test == 2).astype(np.int)
        np_test_result_sub = np.sum(np_result)
        acc_1 = np.sum(np_output_AC)
        acc_2 = np.sum(targets_td_ac)
        acc_rate1 = np_test_result_sub/acc_1
        acc_rate2 = np_test_result_sub/acc_2
        ACC_final = (acc_rate1+acc_rate2)/2
        li_AC.append(ACC_final)

        out_x_sub, out_y_sub = np.where(np_output_sub[0] == np.max(np_output_sub[0]))
        out_x_sub = round(np.mean(out_x_sub))
        out_y_sub = round(np.mean(out_y_sub))

        targets_x, targets_y = np.where(targets_td == 0)
        targets_x = round(np.mean(targets_x))
        targets_y = round(np.mean(targets_y))

        ignition_distance_sub = round(m.sqrt((out_x_sub - targets_x) ** 2 + (out_y_sub - targets_y) ** 2), 2)
        ide_N_sub = ignition_distance_sub/pere_number

        ide_list_sub[i] = ignition_distance_sub
        li_ide.append(ignition_distance_sub)
        li_ide_N.append(ide_N_sub)

        print('--------')
        print('output fire spot', out_x_sub,out_y_sub)
        print('target fire spot', targets_x, targets_y)
        print('perimeter size is',pere_number)
        print('ide_sub is', ignition_distance_sub)
        print('ide_N is ',ide_N_sub)
        print('Acc is ',ACC_final)
        print('--------')



    print("----++++++------")
    print("The average miou of all images is: {:.4f}".format(np.mean(mious_list)))
    print("The average accuracy of all images is: {:.4f}".format(np.mean(accuracy_list)))
    print("The average f1-score of all images is: {:.4f}".format(np.mean(f1_list)))
    print("The average mte of all images is: {:.4f}".format(np.mean(time_mte_list)))

    print("The average sub_ide of all images is: {:.4f}".format(np.mean(ide_list_sub)))
    print("----++++++------")
    print(li_ide)
    print(li_ide_N)
    print(li_AC)

    result_excel = pd.DataFrame()
    result_excel["ide"] = li_ide
    result_excel["ide_N"] = li_ide_N
    result_excel["Acc"] = li_AC
    result_excel.to_excel('data_save.xlsx')
    end_time = time.time()
    print("The average time for each map is: {:.4f} s".format((end_time - start_time) / len(paths_list)))
    li_len = np.arange(1, len(paths_list) + 1).astype(dtype=np.str)
    print(li_len)

    plt.figure(figsize=(20,20))
    plt.bar(li_len, ide_list)
    plt.xlabel('test_map/pic')
    plt.ylabel('distance to real point/m')
    plt.title('ide_show')
    plt.xticks(rotation = 60)
    plt.show()

    return mious_list, accuracy_list, f1_list, time_mte_list, ide_list


if __name__ == '__main__':

    mious_list, accuracy_list, f1_list, time_mte_list, ide_list = main()
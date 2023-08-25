import argparse
import matplotlib.pyplot as plt
import time
import datetime
import os
import shutil
import pandas as pd
# import sys
import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import math as m
from data_util_forsegformer import data_loader
from network.segformer import SegFormer
from network.SCNet import SCNet
from network.DirNet import DIRNet
from loss.loss import miou_loss, smooth_L1_loss, mse_loss, l1_loss  # ,gradual_layer_loss
import torch.nn.functional as F
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='FireNet With Pytorch')

    parser.add_argument('--model', type=str, default='segwind_10pixel',
                        choices=['fcn32s', ' '],
                        help='model name (default: fcn32s)')

    parser.add_argument('--backbone', type=str, default='360',
                        choices=['vgg16', 'resnet18'],
                        help='backbone name (default: vgg16)')

    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')

    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')

    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')

    parser.add_argument('--epochs', type=int, default=550, metavar='N',
                        help='number of epochs to train (default: 50)')

    parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')

    parser.add_argument('--weight-decay', type=float, default=5e-2, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    parser.add_argument('--save-dir', default='./pth/GeoMAC/Segformer_essay/',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--save-pth-iteration', type=int, default=2000,
                        help='save model with n iterations')

    parser.add_argument('--print-iteration', type=int, default=5,
                        help='print train parameters with n iterations')

    parser.add_argument('--log-dir', default='./logs/',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--skip-val', action='store_true', default=True,
                        help='skip validation during training')

    parser.add_argument('--data-path', type=str, default='./data_test/dataset/result/',
                        help='the data path for train')
    parser.add_argument('--path-model', type=str,
                        default='./pth/GeoMAC/210915_exp120_4000x10samples_dv3_res50_l1_R2_5/segnew_resnet_best_model.pth',
                        help='the model path for train')

    args = parser.parse_args()

    return args


class Trainer(object):
    def __init__(self, args):

        self.args = args

        self.device = torch.device(args.device)

        self.train_dataset = data_loader()
        self.valid_dataset = data_loader(args.data_path)
        self.path_list = self.valid_dataset.paths_list

        args.iters_per_epoch = len(self.train_dataset) // (args.num_gpus * args.batch_size)

        args.max_iters = args.epochs * args.iters_per_epoch


        shuffle = True
        if shuffle:
            train_sampler = data.sampler.RandomSampler(self.train_dataset)
        else:
            train_sampler = data.sampler.SequentialSampler(self.train_dataset)


        train_batch_sampler = data.sampler.BatchSampler(train_sampler,
                                                        args.batch_size,
                                                        drop_last=True)

        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.model = SegFormer('MiT-B4').to(self.device)
        state_dic = self.model.state_dict()
        model_dic={}
        pretrained_dic = torch.load('segformer.b4.512x512.ade.160k.pth')['state_dict']
        for k,v in pretrained_dic.items():
            if(k in state_dic and v.shape == state_dic[k].shape):

                model_dic[k] = v
                print('load weights',k)
        state_dic.update(model_dic)
        self.model.load_state_dict(state_dic)

        print(self.model)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

    def train(self):

        epoch, epochs, max_iters = self.args.start_epoch, self.args.epochs, \
                                   self.args.max_iters

        start_time = time.time()

        ide_list = np.zeros(len(self.path_list))
        iteration = 0
        loss_best = 1e3
        loss_rate = 1e3

        np_loss = np.array([])
        li_loss = []
        li_distance = []

        ide_best = 500

        while epoch < epochs:
            self.model.train()

            epoch += 1

            for index, (inputs, targets) in enumerate(self.train_loader):

                iteration += 1

                fire_ = inputs[0]
                fire = fire_.to(self.device)  # It

                target_td_ = targets[0]
                targets_td = target_td_.data.cpu().numpy()

                targets_td = targets_td[0, :, :]

                index1, index2 = np.where(targets_td == np.min(targets_td))
                index1 = round(np.mean(index1))
                index2 = round(np.mean(index2))

                target_td_ = target_td_.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(fire)

                outputs = F.log_softmax(outputs,dim=1)
                criterion = nn.NLLLoss()
                np_outputs = outputs.data.cpu().numpy()

                np_output = np_outputs[0, 0, :, :]


                index = np.where(np_output == np.max(np_output))
                x_0 = index[0]
                fire_spot_x = round(np.mean(x_0))
                y_0 = index[1]
                fire_spot_y = round(np.mean(y_0))
                ignition_distance = round(m.sqrt((fire_spot_x - index1) ** 2 + (fire_spot_y - index2) ** 2), 2)
                li_distance.append(ignition_distance)




                loss = criterion(outputs, target_td_.long())


                loss.backward()


                # for name, parms in self.model.named_parameters():
                #     print('-->name:', name,  '-->grad_value:',parms.grad*10)


                self.optimizer.step()


                eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


                np_loss = np.append(np_loss, loss.item())




                if (iteration % self.args.print_iteration == 0):

                    loss_cur = np.mean(np_loss)
                    distance = round(np.mean(li_distance), 2)

                    li_loss.append(loss_cur)

                    print(
                        "Iters: {:d}/{:d} | Lr: {:.7f} | LossT: {:.6f} | CT: {} | ET: {} | ide: {} ".format(
                            iteration, max_iters, self.optimizer.param_groups[0]['lr'],
                            loss_cur, str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            eta_string,distance))

                    np_loss = np.array([])
                    li_distance = []


                if (iteration % self.args.save_pth_iteration == 0):

                    is_best = False
                    if loss_cur < loss_best:

                        loss_best = loss_cur
                        is_best = True

                    self.save_pth(self.model, self.args, iteration, is_best=is_best, valid_best=False)


            self.lr_scheduler.step()


            self.model.eval()

            for i in range(len(self.path_list)):
                inputs_valid, targets_valid = self.valid_dataset.__getitem__(i)
                fire_valid, landuse_valid, dem_valid, TH_valid, wind_valid = inputs_valid
                fire_valid = fire_valid.unsqueeze(0).to(self.device)
                landuse_valid = landuse_valid.unsqueeze(0).to(self.device)
                dem_valid = dem_valid.unsqueeze(0).to(self.device)
                TH_valid = TH_valid.unsqueeze(0).to(self.device)
                wind_valid = wind_valid.unsqueeze(0).to(self.device)

                outputs_valid= self.model([fire_valid, landuse_valid, dem_valid, TH_valid, wind_valid])

                outputs_valid = F.softmax(outputs_valid, dim=1)
                targets_td_valid = targets_valid[0]
                targets_td_valid = targets_td_valid.data.cpu().numpy()
                np_outputs_valid = outputs_valid.data.cpu().numpy()

                np_output_valid = np_outputs_valid[0, :, :, :]
                for k in range(np_output_valid.shape[-2]):
                    for o in range(np_output_valid.shape[-1]):
                        if(np_output_valid[0,k,o] == np.max(np_output_valid[:,k,o])):
                            np_output_valid[0,k,o] = 1
                out_x_v, out_y_v = np.where(np_output_valid[0] == np.max(np_output_valid[0]))
                out_x_v = round(np.mean(out_x_v))
                out_y_v = round(np.mean(out_y_v))
                targets_x_v, targets_y_v = np.where(targets_td_valid == 0)
                targets_x_v = round(np.mean(targets_x_v))
                targets_y_v = round(np.mean(targets_y_v))
                ignition_distance_valid = round(m.sqrt((out_x_v - targets_x_v) ** 2 + (out_y_v - targets_y_v) ** 2), 2)
                ide_list[i] = ignition_distance_valid
            print("The average ide of all images is: {:.4f}".format(np.mean(ide_list)))

            if(np.mean(ide_list) < ide_best):
                ide_best = np.mean(ide_list)
                valid_best = True
                self.save_pth(self.model, self.args, iteration, is_best=False, valid_best=valid_best, Accindex=round(ide_best,2))

        self.save_pth(self.model, self.args, iteration, is_best=False, valid_best=False)

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        print(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))
        plt.plot(li_loss)
        plt.show()
        savedata = pd.DataFrame(li_loss)
        savedata.to_excel(r'loss_window_segformer.xls')


    def save_pth(self, model, args, iteration, is_best=False, valid_best = False, Accindex=float(0)):

        directory = os.path.expanduser(args.save_dir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = '{}_{}_{}.pth'.format(args.model, args.backbone, str(iteration))
        filename = os.path.join(directory, filename)

        dict_model = {
            'Details': "model dict for fire CANet",
            'iteration': iteration,
            'state_dict_CANet': model.state_dict(),
        }
        torch.save(dict_model, filename)


        if is_best:
            best_filename = '{}_{}_best_model.pth'.format(args.model, args.backbone)
            best_filename = os.path.join(directory, best_filename)
            shutil.copyfile(filename, best_filename)
        if valid_best:
            valid_filename = '{}_{}_best_valid_model_{}.pth'.format(args.model, args.backbone, Accindex)
            best_filename = os.path.join(directory, valid_filename)
            shutil.copyfile(filename, best_filename)


if __name__ == '__main__':

    args = parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    trainer = Trainer(args)
    trainer.train()

    torch.cuda.empty_cache()
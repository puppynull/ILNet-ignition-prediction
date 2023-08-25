import torch
from torch import Tensor
from torch.nn import functional as F
from .base_out import BaseModel
from .heads import SegFormerHead
import numpy as np
import torch.nn as nn
# class SegFormer(BaseModel):
#     def __init__(self, backbone: str = 'MiT-B4', num_classes: int = 4) -> None:
#         super().__init__(backbone, num_classes)
#         self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
#         self.apply(self._init_weights)
#         # self.apply(self.init_pretrained)
#
#     def forward(self, x: Tensor) -> Tensor:
#         y = self.backbone(x)
#         y = self.decode_head(y)   # 4x reduction in image size
#         y = F.interpolate(y, size=x[0].shape[-2:], mode='bilinear', align_corners=False)    # to original image shape
#         return y
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=16,
                            kernel_size=64,
                            stride=16,
                            padding=0),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 32, 3, 1, 0),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, 3, 1, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(320, 128)
        self.mlp2 = torch.nn.Linear(128, 32)

        self.conv5 = torch.nn.Sequential(
            torch.nn.Linear(32, 25),
            nn.Softmax(dim=1))  # nn.Softmax()作为最后一层

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = self.conv5(x)
        return x
class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B4', num_classes: int = 4) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)
        # self.subNet = CNNnet()
        # self.apply(self.init_pretrained)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        # print(y_mid.shape)
        y = F.interpolate(y, size=x[0].shape[-2:], mode='bilinear', align_corners=False)    # to original image shape
        # print(y[:,0,:,:].shape)
        # a = np.zeros([y.shape[0],2])
        # y_1 = y.data.cpu().numpy()
        # for i in range(a.shape[0]):
        #     index_x,index_y = np.where(y_1[i,0,:,:] == np.max(y_1[i,0,:,:]))
        #     index_x = np.mean(index_x)
        #     index_y = np.mean(index_y)
        #     a[i][0] = int(index_x//4)
        #     a[i][1] = int(index_y//4)
        # # print(a)
        # finegrin_feature = torch.zeros([y.shape[0],y_mid.shape[1]]).to('cuda')
        # for i in range(finegrin_feature.shape[0]):
        #     finegrin_feature[i] = y_mid[i,:,a[i][0].astype(np.int),a[i][1].astype(np.int)]
        # finegrin_feature = finegrin_feature.unsqueeze(1)
        # # print(finegrin_feature.shape)
        # sub_y = self.subNet(finegrin_feature)
        return y

if __name__ == '__main__':

    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    device = 'cuda'
    # x = torch.randn(4, 4, 512, 512).to(device)

    model = SegFormer('MiT-B5').to(device)
    img = torch.randn(4, 1, 256, 256).to(device)
    img1 = torch.randn(4, 25, 256, 256).to(device)
    img2 = torch.randn(4, 1, 256, 256).to(device)
    img3 = torch.randn(4, 2).to(device)
    img4 = torch.randn(4, 2).to(device)
    img5 = torch.randn(4, 25, 256, 256).to(device)
    img6 = [img, img1, img2, img3, img4, img5]
    y = model(img6)
    print(y.shape)

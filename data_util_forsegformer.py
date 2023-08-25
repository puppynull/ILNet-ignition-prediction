import os
import random
import numpy as np
import torch
import gdal
from xml.dom.minidom import parse
from scipy import interpolate
import math as m
import cv2
import torch.nn.functional as F

class data_loader(object):
    def __init__(self, path='data_test/dataset/result/', fix_land_dem=True):
        super(data_loader, self).__init__()
        self.path_root = path
        self.fix_land_dem = fix_land_dem
        self.geo_projection = []
        self.geo_transform = []
        self.time_interval = 60
        self.time_random = 0
        self.landtype = 37
        self.code_type = []
        self.code_list = []
        self.combusible_list = []
        self.paths_list = self.get_tif_paths(path)

    def get_tif_paths(self, file_dir='data/dataset/result/', file_format='.tif'):

        file_path_list = []

        for root, dirs, files in os.walk(file_dir):

            if len(dirs) == 0:
                file_path_list.append(root)

        return file_path_list

    def save_tif_data(self, path, data):

        h, w = data.shape

        driver = gdal.GetDriverByName('GTiff')

        dataset = driver.Create(path, w, h, 1, gdal.GDT_Float64)

        dataset.SetProjection(self.geo_projection)
        dataset.SetGeoTransform(self.geo_transform)

        dataset.GetRasterBand(1).WriteArray(data)

        del dataset

    def gaussF(self,x, delta=0.15, u=0):
        p = (m.exp(-1 * ((x - u) ** 2) / (2 * delta))) / (m.sqrt(2 * m.pi) * m.sqrt(delta))
        return p

    def distance_to_gauss(self,img_test):
        d_li = []
        for i in range(img_test.shape[0]):
            for j in range(img_test.shape[1]):
                d = round(m.sqrt((i - 50) ** 2 + (j - 50) ** 2), 2)
                d_li.append(d)
        return d_li

    def label_smoothing(self,path=""):
        data_fire=gdal.Open(path)
        fire_w=data_fire.RasterXSize
        fire_h=data_fire.RasterYSize
        fire_band=data_fire.GetRasterBand(1)
        data_raster_fire = fire_band.ReadAsArray(0, 0, fire_w, fire_h)
        data_raster_fire[data_raster_fire<-100] = 200
        index_fire = (data_raster_fire == np.min(data_raster_fire))
        label = index_fire.astype(np.float)
        a,b=np.where(label==np.max(label))


        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                d = round(m.sqrt((i - a) ** 2 + (j - b) ** 2), 2) / +65

                label[i, j] = self.gaussF(d, 0.15, 0)

        label_index = np.max(label)
        label = label/label_index

        del data_fire
        return label
    def get_tif_data(self, path="", geo=True, fliter=False):

        dataset = gdal.Open(path)
        if dataset == None:
            print(path + " load file error!")

        if geo:
            self.geo_projection = dataset.GetProjection()
            self.geo_transform = dataset.GetGeoTransform()

        data_w = dataset.RasterXSize
        data_h = dataset.RasterYSize


        band = dataset.GetRasterBand(1)

        data_raster = band.ReadAsArray(0, 0, data_w, data_h)

        if fliter:
            index = data_raster > 1e5

            data_raster[index] = 0

        del dataset

        return data_raster.astype(np.float32)


    def get_tif_data_fire(self, path="", geo=False, fliter=False):

        dataset = gdal.Open(path)
        if dataset == None:
            print(path + " load file error!")

        if geo:
            self.geo_projection = dataset.GetProjection()
            self.geo_transform = dataset.GetGeoTransform()

        data_w = dataset.RasterXSize
        data_h = dataset.RasterYSize

        band = dataset.GetRasterBand(1)

        data_raster = band.ReadAsArray(0, 0, data_w, data_h)

        if fliter:
            index = data_raster >= 0
            index1= data_raster < 0
            data_raster[index] = 1
            data_raster[index1] = 0
        del dataset

        return data_raster.astype(np.float32)


    def get_png_data(self,path=''):
        # img=cv2.imread(path)
        return cv2.imread(path)/255.0

    def get_xml_data(self, path, length=1):

        DOMTree = parse(path)

        root = DOMTree.documentElement

        environment = root.getElementsByTagName("Environment")

        meteorology = np.ones(4)  #
        norm_para = np.array([10.0, 1.0, 20.0, 100])

        self.time_interval = np.array(environment[0].getElementsByTagName(
            "TimeInterval")[0].getElementsByTagName("Value")[0].childNodes[0].nodeValue, np.float32)  # minute

        str_windspeed = environment[0].getElementsByTagName(
            "WindSpeed")[0].getElementsByTagName("Value")[0].childNodes[0].nodeValue
        str_winddirection = environment[0].getElementsByTagName(
            "WindDirection")[0].getElementsByTagName("Value")[0].childNodes[0].nodeValue
        str_temperature = environment[0].getElementsByTagName(
            "Temperature")[0].getElementsByTagName("Value")[0].childNodes[0].nodeValue
        str_humidity = environment[0].getElementsByTagName(
            "Humidity")[0].getElementsByTagName("Value")[0].childNodes[0].nodeValue

        meteorology[0] = np.array(str_windspeed, np.float32)

        meteorology[1] = np.array(str_winddirection, np.float32)
        meteorology[2] = np.array(str_temperature, np.float32)

        meteorology[3] = np.array(str_humidity, np.float32)

        return meteorology / norm_para

    def get_data_from_str(self, str_, length):

        num = len(str_)
        data = np.zeros((num, length))

        for i in range(num):

            str_i = str_[i].split(" ")
            data_i = np.zeros((1, len(str_i)))
            x = np.linspace(0, len(str_i) - 1, len(str_i))

            for j in range(len(str_i)):

                data_i[0, j] = float(str_i[j])

            data[i, :] = interpolate.interp1d(x, data_i, kind='linear')

    def get_from_str_list(self, string):

        str_list = string.split()

        float_list = list(map(float, str_list))

        return np.array(float_list)

    def get_VET(self, path):

        DOMTree = parse(path)

        root = DOMTree.documentElement

        parameters = root.getElementsByTagName("land_information")

        parameters_t = parameters[0].getElementsByTagName("type")

        str_landValue = parameters_t[0].getElementsByTagName(
            "land_value")[0].childNodes[0].nodeValue
        LV = self.get_from_str_list(str_landValue)

        str_landCoefficient = parameters_t[0].getElementsByTagName(
            "land_coefficient")[0].childNodes[0].nodeValue
        LC = self.get_from_str_list(str_landCoefficient)

        str_landType = parameters_t[0].getElementsByTagName(
            "land_type")[0].childNodes[0].nodeValue
        LT = self.get_from_str_list(str_landType)

        return LV, LC, LT

    def evt_setting(self):

        code_list, LC, combusible_list = self.get_VET(self.path_root + 'Model_Internal_Parameter.xml')

        c_type = np.unique(LC)
        code_reclassfy = np.zeros(code_list.shape)

        for i in range(len(c_type)):

            index = (LC == c_type[i])
            code_reclassfy[index] = i


        self.code_type = code_reclassfy
        self.code_list = code_list
        self.combusible_list = combusible_list

    def vet_encoder(self, landuse):

        code_num = int(self.code_type.max() + 1)

        landuse_ = np.zeros((code_num, landuse.shape[0], landuse.shape[1]))
        land_mask = np.zeros_like(landuse)

        for i in range(len(self.code_list)):

            index = (landuse == self.code_list[i])

            layer_index = int(self.code_type[i])

            landuse_[layer_index] += index.astype(np.int)

            if self.combusible_list[i] == 1:
                land_mask[index] = 1.0

        return landuse_, land_mask

    def landuse_encoder(self, landuse):

        code_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 255])
        combusible_list = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 0, 0, 0, 0, 1, 0, 0, 0])

        code_num = len(code_list)
        onehot = np.eye(code_num)

        landuse_ = np.zeros((code_num, landuse.shape[0], landuse.shape[1]))
        land_mask = np.zeros_like(landuse)

        for i in range(len(code_list)):

            index = (landuse == code_list[i]).astype(np.int)

            landuse_[i] += index

            if combusible_list[i] == 1:
                land_mask[index] = 1.0

        return landuse_, land_mask
    def another_landuse_encoder(self, landuse):
        landuse[landuse == 255] = 25

        return (landuse-13)/13

    def get_fire_label(self, path="", geo=False):

        dataset = gdal.Open(path)
        if dataset == None:
            print(path + " load file error!")

        if geo:
            self.geo_projection = dataset.GetProjection()
            self.geo_transform = dataset.GetGeoTransform()

        data_w = dataset.RasterXSize
        data_h = dataset.RasterYSize

        band = dataset.GetRasterBand(1)

        data_raster = band.ReadAsArray(0, 0, data_w, data_h)

        for i in range(data_raster.shape[0]):
            for j in range(data_raster.shape[1]):
                if(data_raster[i][j]>400):
                    data_raster[i][j] = 3

                elif(160< data_raster[i][j] ):
                    data_raster[i][j] = 2

                elif(0 < data_raster[i][j]<= 160 ):
                    data_raster[i][j] = 1


        ing1,ing2 = np.where(data_raster == 0)
        data_raster[ing1[0]-10:ing1[0]+11,ing2[0]-10:ing2[0]+11] = 0


        return data_raster.astype(np.float32)

    def gen_wind(self, windforce, angle, grid):
        wind_ = torch.zeros((1, 3, 3))
        wind_vector = np.zeros((grid.shape[0],grid.shape[1]))
        angle1 = np.array(angle * np.pi / 180)
        angle2 = np.array((angle - 45) * np.pi / 180)
        angle1 = torch.from_numpy(angle1)
        angle2 = torch.from_numpy(angle2)
        wind_[:, 0, 1] = torch.sin(angle1)*windforce
        wind_[:, 1, 2] = torch.cos(angle1)*windforce
        wind_[:, 2, 1] = -wind_[:, 0, 1]
        wind_[:, 1, 0] = -wind_[:, 1, 2]

        wind_[:, 0, 2] = torch.cos(angle2)*windforce
        wind_[:, 0, 0] = torch.sin(angle2)*windforce
        wind_[:, 2, 0] = -wind_[:, 0, 2]
        wind_[:, 2, 2] = -wind_[:, 0, 0]
        wind_[:, 1, 1] = windforce
        index1, index2 = np.where(grid == 1)
        index_max_x = np.max(index1)
        index_min_x = np.min(index1)
        index_max_y = np.max(index2)
        index_min_y = np.min(index2)
        wind_ = wind_.unsqueeze(1)
        wind_big = F.interpolate(wind_, [index_max_x-index_min_x+1, index_max_y-index_min_y+1], mode='bilinear', align_corners=True)
        wind_big = wind_big.squeeze(0).squeeze(0)
        wind_big = wind_big.numpy()

        wind_vector[index_min_x:index_max_x+1,index_min_y:index_max_y+1] = wind_big
        wind_vector[grid == 0] = 0
        return wind_vector

    def __getitem__(self, index):

        path_root = self.paths_list[index]

        path_fix =path_root.split("\\")[0].replace("result/", "slices/")

        landuse = self.get_tif_data(path_fix + '/Landuse_3857.tif', False, True)

        dem_norm = 100
        dem = self.get_tif_data(path_fix + '/DEM_3857.tif', True, True)
        dem = (dem - np.mean(dem)) / dem_norm
        landuse[landuse==255] = 25
        landuse = (landuse-13)/25



        inputs = self.get_tif_data_fire(path_root + '/fire_0360.tif')


        targets = self.get_fire_label(path_root + '/fire_time.tif')
        meteorology = self.get_xml_data(path_root + '/meteorology.xml')
        t_wind = meteorology[[0, 1]]

        wind_vector = self.gen_wind(t_wind[0], t_wind[1], inputs)

        t_inputs = np.zeros((4,inputs.shape[0],inputs.shape[1]))
        t_inputs[0,:,:] = inputs
        t_inputs[1,:,:] = dem
        t_inputs[2,:,:] = landuse
        t_inputs[3,:,:] = wind_vector

        t_inputs = torch.from_numpy(t_inputs).float()

        t_targets = torch.from_numpy(targets).float()

        return [t_inputs], [t_targets]

    def get_time_random(self):
        #
        return self.time_random

    def get_geo_projection(self):

        return self.geo_projection

    def get_geo_transform(self):

        return self.geo_transform

    def __len__(self):
        return len(self.paths_list)



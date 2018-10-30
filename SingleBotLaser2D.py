import numpy as np
import random
import utils
import cv2
import math

class SingleBotLaser2Dgrid:
    def __init__(self, bot_pos, bot_param, fname):
        self.bot_pos = bot_pos
        self.bot_param = bot_param
        self.img_map = utils.Image2Map(fname)

        scale = 1
        img = utils.Image2Map(fname)
        img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
        self.img_map = img
    
    def BotAction(self, aid):
        vec = [np.sin(np.deg2rad(self.bot_pos[2])), np.cos(np.deg2rad(self.bot_pos[2]))]
        vel = self.bot_param[4]
        ang = self.bot_param[5]

        if aid == 1:
            self.bot_pos[0] -= vel*vec[0]
            self.bot_pos[1] += vel*vec[1]
        if aid == 2:
            self.bot_pos[0] += vel*vec[0]
            self.bot_pos[1] -= vel*vec[1]
        if aid == 3:
            self.bot_pos[2] -= ang
            self.bot_pos[2] = self.bot_pos[2] % 360
        if aid == 4:  
            self.bot_pos[2] += ang
            self.bot_pos[2] = self.bot_pos[2] % 360
        
        if aid == 5:
            self.bot_pos[1] -= vel
        if aid == 6:
            self.bot_pos[0] -= vel
        if aid == 7:
            self.bot_pos[0] += vel
        if aid == 8:
            self.bot_pos[1] += vel
        
        sig=[0.5,0.5,0.5]
        self.bot_pos[0] += random.gauss(0,sig[0])
        self.bot_pos[1] += random.gauss(0,sig[1])
        self.bot_pos[2] += random.gauss(0,sig[2])

    def Sensor(self):
        sense_data = []
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(self.bot_param[0]):
            theta = self.bot_pos[2] + self.bot_param[1] + i*inter
            sense_data.append(self.RayCast(np.array((self.bot_pos[0], self.bot_pos[1])), theta))
        return sense_data

    def RayCast(self, pos, theta):
        end = np.array((pos[0] + self.bot_param[3]*np.cos(np.deg2rad(theta)), pos[1] + self.bot_param[3]*np.sin(np.deg2rad(theta))))

        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = utils.Bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.bot_param[3]
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if self.img_map[p[1], p[0]] < 0.6:
                tmp = math.pow(float(p[0]) - pos[0], 2) + math.pow(float(p[1]) - pos[1], 2)
                tmp = math.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist
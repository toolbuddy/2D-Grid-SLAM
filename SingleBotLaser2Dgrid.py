import numpy as np
import random
import utils
import cv2
import math
from GridMap import *
from ParticleFilter import *
import copy
import matplotlib.pyplot as plt
import Icp2d

class SingleBotLaser2Dgrid:
    def __init__(self, bot_pos, bot_param, fname):
        self.bot_pos = bot_pos
        self.bot_param = bot_param
        self.img_map = self.Image2Map(fname)

        scale = 1
        img = self.Image2Map(fname)
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
        end = np.array((pos[0] + bot_param[3]*np.cos(np.deg2rad(theta)), pos[1] + bot_param[3]*np.sin(np.deg2rad(theta))))

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

    def Image2Map(self, fname):
        im = cv2.imread(fname)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m

def Map2Image(m):
    img = (255*m).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def Draw(img_map, scale, bot_pos, sensor_data, bot_param):
    img = img_map.copy()
    img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
    img = Map2Image(img)
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    for pts in plist:
        cv2.line(
            img, 
            (int(scale*bot_pos[0]), int(scale*bot_pos[1])), 
            (int(scale*pts[0]), int(scale*pts[1])),
            (255,0,0), 1)

    cv2.circle(img,(int(scale*bot_pos[0]), int(scale*bot_pos[1])), int(3*scale), (0,0,255), -1)
    return img

def DrawParticle(img, plist, scale=1.0):
    for p in plist:
        cv2.circle(img,(int(scale*p.pos[0]), int(scale*p.pos[1])), int(2), (0,200,0), -1)
    return img

def SensorMapping(m, bot_pos, bot_param, sensor_data):
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    for i in range(bot_param[0]):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        theta = bot_pos[2] + bot_param[1] + i*inter
        m.GridMapLine(
        int(bot_pos[0]), 
        int(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
        int(bot_pos[1]),
        int(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta)))
        )

def AdaptiveGetMap(gmap):
    mimg = gmap.GetMapProb(
        gmap.boundary[0]-20, gmap.boundary[1]+20, 
        gmap.boundary[2]-20, gmap.boundary[3]+20 )
    #mimg = gmap.GetMapProb(0,500,0,500)
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg

def SensorData2PointCloud(sensor_data, bot_pos, bot_param):
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    tmp = []
    for i in range(len(sensor_data)):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        tmp.append(plist[i])
    tmp = np.array(tmp)
    return tmp

def DrawAlign(Xc, Pc, R, T):
    if Xc.shape[0] < 20 or Pc.shape[0] < 20:
        return 255*np.ones((100,100), np.uint8)
    shift = 10.0
    Xc_ = Icp2d.Transform(xc, R, T)
    max_x = np.max(np.array([np.max(Xc_[:,0]), np.max(Pc[:,0])]) )
    max_y = np.max(np.array([np.max(Xc_[:,1]), np.max(Pc[:,1])]) )
    min_x = np.min(np.array([np.min(Xc_[:,0]), np.min(Pc[:,0])]) )
    min_y = np.min(np.array([np.min(Xc_[:,1]), np.min(Pc[:,1])]) )
    img = 255*np.ones((int(max_y-min_y+2*shift),int(max_x-min_x+2*shift),3), np.uint8)
    for i in range(Xc_.shape[0]):
        cv2.circle(img, (int(Xc[i,0]-min_x+shift), int(Xc[i,1]-min_y+shift)), int(2), (200,200,200), -1)
    for i in range(Xc.shape[0]):
        cv2.circle(img, (int(Xc_[i,0]-min_x+shift), int(Xc_[i,1]-min_y+shift)), int(2), (0,0,255), -1)
    for i in range(Pc.shape[0]):
        cv2.circle(img, (int(Pc[i,0]-min_x+shift), int(Pc[i,1]-min_y+shift)), int(2), (255,0,0), -1)
    return img
 
def Rotation2Deg(R):
    cos = R[0,0]
    sin = R[1,0]
    theta = np.rad2deg(np.arccos(np.abs(cos)))
    
    if cos>0 and sin>0:
        return theta
    elif cos<0 and sin>0:
        return 180-theta
    elif cos<0 and sin<0:
        return 180+theta
    elif cos>0 and sin<0:
        return 360-theta
    elif cos==0 and sin>0:
        return 90.0
    elif cos==0 and sin<0:
        return 270.0
    elif cos>0 and sin==0:
        return 0.0
    elif cos<0 and sin==0:
        return 180.0

if __name__ == '__main__':
    # Initialize OpenCV Windows
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)

    # Initialize 2D Environment
    # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
    bot_param = [240,-30.0, 210.0, 150.0, 6.0, 6.0]
    bot_pos = np.array([150.0, 100.0, 0.0])
    env = SingleBotLaser2Dgrid(bot_pos, bot_param, 'map_large.png')

    # Initialize GridMap
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [0.4, -0.4, 5.0, -5.0] 
    m = GridMap(map_param, gsize=1.0)
    sensor_data = env.Sensor()
    SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)

    img = Draw(env.img_map, 1, env.bot_pos, sensor_data, env.bot_param)
    mimg = AdaptiveGetMap(m)
    cv2.imshow('view',img)
    cv2.imshow('map',mimg)

    # Initialize Particle
    pf = ParticleFilter(bot_pos.copy(), bot_param, copy.deepcopy(m), 10)
    sensor_data_rec = sensor_data.copy()
    
    # Scan Matching Test
    matching_m = GridMap(map_param, gsize=1.0)
    SensorMapping(matching_m, env.bot_pos, env.bot_param, sensor_data)
    matching_pos = np.array([150.0, 100.0, 0.0])

    # Main Loop
    while(1):
        # Input Control
        action = -1
        k = cv2.waitKey(1)
        if k==ord('w'):
            action = 1
        if k==ord('s'):
            action = 2
        if k==ord('a'):
            action = 3
        if k==ord('d'): 
            action = 4 
        
        if k==ord('i'):
            action = 5
        if k==ord('j'):
            action = 6
        if k==ord('l'):
            action = 7
        if k==ord('k'):
            action = 8
        
        if action > 0:
            env.BotAction(action)
            sensor_data = env.Sensor()
            SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)
    
            img = Draw(env.img_map, 1, env.bot_pos, sensor_data, env.bot_param)
            mimg = AdaptiveGetMap(m)
            
            pf.Feed(action, sensor_data)
            mid = np.argmax(pf.weights)
            imgp0 = AdaptiveGetMap(pf.particle_list[mid].gmap)
            
            img = DrawParticle(img, pf.particle_list)
            cv2.imshow('view',img)
            cv2.imshow('map',mimg)

            cv2.imshow('particle_map',imgp0)
            pf.Resampling(sensor_data)

            pc = SensorData2PointCloud(sensor_data_rec, env.bot_pos, env.bot_param)
            xc = SensorData2PointCloud(sensor_data, env.bot_pos, env.bot_param)
            R,T = Icp2d.Icp(300,pc,xc)
            Ttot = np.array([[matching_pos[0], matching_pos[1]]])
            Ttot = Icp2d.Transform(Ttot, R, T)[0]
            theta = matching_pos[2]
            Rtot = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                             [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
            deg = Rotation2Deg(np.matmul(R,Rtot))
            matching_pos = [Ttot[0], Ttot[1], deg]
            aimg = DrawAlign(xc, pc, R, T)
            cv2.imshow('align',aimg)

            SensorMapping(matching_m, matching_pos, env.bot_param, sensor_data)
            matching_img = AdaptiveGetMap(matching_m)
            cv2.imshow('matching_map',matching_img)

            
            sensor_data_rec = sensor_data.copy()

        
    cv2.destroyAllWindows()
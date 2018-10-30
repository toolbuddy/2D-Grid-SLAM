import numpy as np
import random
import utils
import cv2
import math
from GridMap import *
from ParticleFilter import *
from SingleBotLaser2D import *
import copy

def Draw(img_map, scale, bot_pos, sensor_data, bot_param):
    img = img_map.copy()
    img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
    img = utils.Map2Image(img)
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
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg

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

    cv2.destroyAllWindows()
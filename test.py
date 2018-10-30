import numpy as np
import cv2
from SingleBotLaser2D import *
from GridMap import *
from ParticleFilter import *
import utils
import copy

def Draw(bot_pos, line_list, sensor_data, bot_param, scale=5.0, view=(512,512)):
    img = 255*np.ones((512,512,3), np.uint8)
    
    for line in line_list:
        pt1 = np.round(scale * line[0])
        pt2 = np.round(scale * line[1])
        cv2.line(img, tuple(pt1.astype(np.int).tolist()), tuple(pt2.astype(np.int).tolist()), (0,0,0), 3)

    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    for pts in plist:
        cv2.line(
            img, 
            (int(scale*bot_pos[0]), int(scale*bot_pos[1])), 
            (int(scale*pts[0]), int(scale*pts[1])),
            (255,0,0), 1)

    cv2.circle(img,(int(scale*bot_pos[0]), int(scale*bot_pos[1])), int(scale*1.5), (0,0,255), -1)
    return img

def DrawParticle(img, plist, scale=5.0):
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
    
if __name__ == '__main__':
    # Initialize OpenCV Windows
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)

    # Initialize 2D Environment
    # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
    bot_param = [60, -30.0, 210.0, 50.0, 1.0, 3.0]
    bot_pos = np.array([40.0, 40.0, 0.0])
    env = SingleBotLaser2D(bot_pos, bot_param)

    # Obstacle
    env.line_list.append([np.array((10,10)), np.array((10,90))])
    env.line_list.append([np.array((10,10)), np.array((80,10))])
    env.line_list.append([np.array((10,90)), np.array((80,90))])
    env.line_list.append([np.array((80,10)), np.array((80,90))])

    env.line_list.append([np.array((35,10)), np.array((45,30))])
    env.line_list.append([np.array((45,10)), np.array((55,30))])
    env.line_list.append([np.array((45,30)), np.array((55,30))])

    env.line_list.append([np.array((70,50)), np.array((60,90))])
    env.line_list.append([np.array((70,50)), np.array((80,50))])
    
    env.line_list.append([np.array((20,70)), np.array((40,70))])
    env.line_list.append([np.array((20,70)), np.array((20,90))])
    env.line_list.append([np.array((40,70)), np.array((40,90))])

    # Initialize GridMap
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [0.4, -0.4, 5.0, -5.0] 
    m = GridMap(map_param, gsize=0.5)
    sensor_data = env.Sensor()
    SensorMapping(m, env.bot_pos, env.bot_param, sensor_data)

    img = Draw(env.bot_pos, env.line_list, sensor_data, env.bot_param)
    mimg = AdaptiveGetMap(m)
    cv2.imshow('view',img)
    cv2.imshow('map',mimg)

    # Initialize Particle
    pf = ParticleFilter(bot_pos.copy(), bot_param, copy.deepcopy(m), 10)

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
    
            img = Draw(env.bot_pos, env.line_list, sensor_data, env.bot_param)
            mimg = AdaptiveGetMap(m)
            
            pf.Feed(action, sensor_data)
            mid = np.argmax(pf.weights)
            imgp0 = AdaptiveGetMap(pf.particle_list[mid].gmap)
            
            img = DrawParticle(img, pf.particle_list)
            cv2.imshow('view',img)
            cv2.imshow('map',mimg)
            cv2.imshow('p0_map',imgp0)
            pf.Resampling(sensor_data)
        
    cv2.destroyAllWindows()
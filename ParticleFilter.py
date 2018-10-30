import numpy as np
from GridMap import *
import random
import math
import utils
import copy
import threading

class Particle:
    def __init__(self, pos, bot_param, gmap):
        self.pos = pos
        self.bot_param = bot_param
        self.gmap = gmap

    def Sampling(self, aid, sig=[0.4,0.4,0.4]):
        vec = [np.sin(np.deg2rad(self.pos[2])), np.cos(np.deg2rad(self.pos[2]))]
        vel = self.bot_param[4]
        ang = self.bot_param[5]

        if aid == 1:
            self.pos[0] -= vel*vec[0]
            self.pos[1] += vel*vec[1]
        if aid == 2:
            self.pos[0] += vel*vec[0]
            self.pos[1] -= vel*vec[1]
        if aid == 3:
            self.pos[2] -= ang
            self.pos[2] = self.pos[2] % 360
        if aid == 4:  
            self.pos[2] += ang
            self.pos[2] = self.pos[2] % 360
        
        if aid == 5:
            self.pos[1] -= vel
        if aid == 6:
            self.pos[0] -= vel
        if aid == 7:
            self.pos[0] += vel
        if aid == 8:
            self.pos[1] += vel

        self.pos[0] += random.gauss(0,sig[0])
        self.pos[1] += random.gauss(0,sig[1])
        self.pos[2] += random.gauss(0,sig[2])

    def NearestDistance(self, x, y, wsize, th):
        min_dist = 9999
        min_x = None
        min_y = None
        gsize = self.gmap.gsize
        xx = int(round(x/gsize))
        yy = int(round(y/gsize))
        for i in range(xx-wsize, xx+wsize):
            for j in range(yy-wsize, yy+wsize):
                if self.gmap.GetGridProb((i,j)) < th:
                    dist = (i-xx)*(i-xx) + (j-yy)*(j-yy)
                    if dist < min_dist:
                        min_dist = dist
                        min_x = i
                        min_y = j

        return math.sqrt(float(min_dist)*gsize)

    def LikelihoodField(self, sensor_data):
        p_hit = 0.9
        p_rand = 0.1
        sig_hit = 3.0
        q = 1
        plist = utils.EndPoint(self.pos, self.bot_param, sensor_data)
        for i in range(len(plist)):
            if sensor_data[i] > self.bot_param[3]-1 or sensor_data[i] < 1:
                continue
            dist = self.NearestDistance(plist[i][0], plist[i][1], 4, 0.2)
            q = q * (p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
            #q += math.log(p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.bot_param[3])
        return q

    def Mapping(self, sensor_data):
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(self.bot_param[0]):
            if sensor_data[i] > self.bot_param[3]-1 or sensor_data[i] < 1:
                continue
            theta = self.pos[2] + self.bot_param[1] + i*inter
            self.gmap.GridMapLine(
            int(self.pos[0]), 
            int(self.pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
            int(self.pos[1]),
            int(self.pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta)))
            )

class ParticleFilter:
    def __init__(self, pos, bot_param, gmap, size):
        self.size = size
        self.particle_list = []
        self.weights = np.ones((size), dtype=float) / size
        p = Particle(pos.copy(), bot_param, copy.deepcopy(gmap))
        for i in range(size):
            self.particle_list.append(copy.deepcopy(p))
    
    def ParticleMapping(plist, sensor_data):
        threads = []
        for p in plist:
            threads.append(threading.Thread(target=p.Mapping, args=(sensor_data,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def Resampling(self, sensor_data):
        map_rec = np.zeros((self.size))
        re_id = np.random.choice(self.size, self.size, p=list(self.weights))
        new_particle_list = []
        for i in range(self.size):
            if map_rec[re_id[i]] == 0:
                self.particle_list[re_id[i]].Mapping(sensor_data)
                map_rec[re_id[i]] = 1
            new_particle_list.append(copy.deepcopy(self.particle_list[re_id[i]]))
        self.particle_list = new_particle_list
        self.weights = np.ones((self.size), dtype=float) / float(self.size)

    def Feed(self, control, sensor_data):
        field = np.zeros((self.size), dtype=float)
        for i in range(self.size):
            self.particle_list[i].Sampling(control)
            field[i] = self.particle_list[i].LikelihoodField(sensor_data)
            #self.particle_list[i].Mapping(sensor_data)

        self.weights = field / np.sum(field)
        #self.Resampling(sensor_data)
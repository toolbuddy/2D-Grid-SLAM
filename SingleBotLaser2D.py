import numpy as np
import random

class SingleBotLaser2D:
    def __init__(self, bot_pos, bot_param):
        self.bot_pos = bot_pos
        self.bot_param = bot_param
        self.line_list = []

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
        p = np.array((pos[0],pos[1]))
        max_dist = self.bot_param[3]
        r = np.array((max_dist*np.cos(np.deg2rad(theta)), max_dist*np.sin(np.deg2rad(theta))))
        dist = np.zeros((len(self.line_list)))
        i = 0
        for line in self.line_list:
            q = np.array(line[0])
            s = np.array(line[1]) - q
            dist[i] = self.Intersection(p,r,q,s)
            i += 1
        return np.min(dist)

    def Intersection(self,p,r,q,s):
        if np.cross(r, s) == 0 and np.cross((q-p), r) == 0:    # collinear
            t0 = np.dot(q-p, r)/np.dot(r, r)
            t1 = t0 + np.dot(s, r)/np.dot(r, r)
            if ((np.dot(s, r) > 0) and (0 <= t1 - t0 <= 1)) or ((np.dot(s, r) <= 0) and (0 <= t0 - t1 <= 1)):
                #print('collinear and overlapping, q_s in p_r')
                return 0.0
            else:
                #print('collinear and disjoint')
                return np.linalg.norm(r)
        elif np.cross(r, s) == 0 and np.cross((q-p), r) != 0:  # parallel r × s = 0 and (q − p) × r ≠ 0,
            #print('parallel')
            return np.linalg.norm(r)
        else:
            t = np.cross((q - p), s) / np.cross(r, s)
            u = np.cross((q - p), r) / np.cross(r, s)
            if 0 <= t <= 1 and 0 <= u <= 1:
                #print('intersection: ', p + t*r)
                return t*np.linalg.norm(r)
            else:
                #print('not parallel and not intersect')
                return np.linalg.norm(r)
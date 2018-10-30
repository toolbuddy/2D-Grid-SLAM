import numpy as np
import utils

class GridMap:
    def __init__(self, map_param, gsize=1.0):
        self.map_param = map_param
        self.gmap = {}
        self.gsize = gsize
        self.boundary = [9999,-9999,9999,-9999]

    def GetGridProb(self, pos):
        if pos in self.gmap:
            return np.exp(self.gmap[pos]) / (1.0 + np.exp(self.gmap[pos]))
        else:
            return 0.5

    def GetCoordProb(self, pos):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        return self.GetGridProb((x,y))

    def GetMapProb(self, x0, x1, y0, y1):
        map_prob = np.zeros((y1-y0, x1-x0))
        idx = 0
        for i in range(x0, x1):
            idy = 0
            for j in range(y0, y1):
                map_prob[idy, idx] = self.GetGridProb((i,j))
                idy += 1
            idx += 1
        return map_prob

    def GridMapLine(self, x0, x1, y0, y1):
        # Scale the position
        x0, x1 = int(round(x0/self.gsize)), int(round(x1/self.gsize))
        y0, y1 = int(round(y0/self.gsize)), int(round(y1/self.gsize))

        rec = utils.Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            if i < len(rec)-2:
                change = self.map_param[0]
            else:
                change = self.map_param[1]

            if rec[i] in self.gmap:
                self.gmap[rec[i]] += change
            else:
                self.gmap[rec[i]] = change
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]

            if self.gmap[rec[i]] > self.map_param[2]:
                self.gmap[rec[i]] = self.map_param[2]
            if self.gmap[rec[i]] < self.map_param[3]:
                self.gmap[rec[i]] = self.map_param[3]
    
if __name__ == '__main__':
    #lo_occ, lo_free, lo_max, lo_min
    map_param = [0.9, -0.7, 5.0, -5.0]
    m = GridMap(map_param)
    pos = (0.0,0.0)
    m.gmap[pos] = 0.1
    print(m.GetProb(pos))
    print(m.GetProb((0,0)))

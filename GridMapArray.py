import numpy as np

class GridMap:
    def __init__(self, map_param, gsize=1.0):
        self.map_param = map_param
        self.gmap = np.zeros((500,500))
        self.gsize = gsize

    def GetGridProb(self, pos):
        value = self.gmap[pos[0],pos[1]]
        return np.exp(value) / (1.0 + np.exp(value))

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

        rec = self.Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            if i < len(rec)-3:
                change = self.map_param[0]
            else:
                change = self.map_param[1]

            self.gmap[rec[i][0],rec[i][1]] += change
            if rec[i] in self.gmap:
                self.gmap[rec[i][0],rec[i][1]] += change

            if self.gmap[rec[i][0],rec[i][1]] > self.map_param[2]:
                self.gmap[rec[i][0],rec[i][1]] = self.map_param[2]
            if self.gmap[rec[i][0],rec[i][1]] < self.map_param[3]:
                self.gmap[rec[i][0],rec[i][1]] = self.map_param[3]
    
    def Bresenham(self, x0, x1, y0, y1):
        rec = []
        "Bresenham's line algorithm"
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                rec.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                rec.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return rec

if __name__ == '__main__':
    #lo_occ, lo_free, lo_max, lo_min
    map_param = [0.9, -0.7, 5.0, -5.0]
    m = GridMap(map_param)
    pos = (0.0,0.0)
    m.gmap[pos] = 0.1
    print(m.GetProb(pos))
    print(m.GetProb((0,0)))
    print(m.Bresenham(0,5,0,3))

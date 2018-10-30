import Icp2d

def SensorData2PointCloud(sensor_data, bot_pos, bot_param):
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    tmp = []
    for i in range(len(sensor_data)):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        tmp.append(plist[i])
    tmp = np.array(tmp)
    return tmp

def Matching(sensor_data, sensor_data_rec):
    pass

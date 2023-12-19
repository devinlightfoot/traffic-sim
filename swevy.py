import random as rand
import numpy as np
from matplotlib import pyplot as plt

# set number of lanes and length of the road
num_lanes = 3
road_length = 1000
empty_road = np.array([[0 for i in range(road_length)] for i in range(num_lanes)])


def populateRoad(n, v_max, road):
    w = len(road)
    l = len(road[0])
    populated = road
    i = 1
    while i <= n:
        rand_x = rand.randint(0, w - 1)
        rand_y = rand.randint(0, l - 1)
        if populated[rand_x][rand_y] == 0:
            populated[rand_x][rand_y] = rand.randint(1, v_max[rand_x] + 1)
            i += 1
    return populated


# set maximum speed per lane of the road
v_max = [6, 5, 4]
# set total number of cars on road
n = 100
road = populateRoad(n, v_max, empty_road)
# probability of  acceleration noise
p_y = 0.25
# probability of lane change
p_x = 0.1
# record initial road configuration
position = np.nonzero(road)
nonZs=[[]]
for i,x in enumerate(position[0]):
    nonZs[0].append([x,position[1][i]])
#nonZs=np.array(nonZs)
#print(nonZs)
t = 0
# set duration of simulation
finish = 2

# implement timestep update loop
while t < finish:
    tmp = road
    carArr = np.nonzero(tmp)
    carOrdinates=[]
    for i,x in enumerate(carArr[0]):
        carOrdinates.append([x,position[1][i]])
    if t > 0:
        nonZs.append(carOrdinates)

    t += 1

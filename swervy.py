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
n = 3
road = populateRoad(n, v_max, empty_road)
# probability of  acceleration noise
p_y = 0.1
# probability of lane change
p_x = 0.1
# record initial road configuration
position = np.nonzero(road)
nonZs = [[]]
for i, x in enumerate(position[0]):
    nonZs[0].append([x, position[1][i]])
# nonZs=np.array(nonZs)
# print(nonZs)
t = 0
# set duration of simulation
finish = 3

# implement timestep update loop
while t < finish:
    tmp = road
    carArr = np.nonzero(tmp)
    carOrdinates = []
    lanes = [[] for i in range(num_lanes)]
    for i, x in enumerate(carArr[0]):
        carOrdinates.append([x, carArr[1][i]])
        lanes[x].append(carArr[1][i])
    if t > 0:
        nonZs.append(carOrdinates)
    print(nonZs[t])
    for i, pos in enumerate(carOrdinates):
        #print("initial")
        #print(tmp[pos[0]][pos[1]]-1)
        # implement NaSch algo for each car
        # will need to implement lane changing updates before lane updates
        # step 1 (is good)
        tmp[pos[0]][pos[1]] = min(tmp[pos[0]][pos[1]] + 1, v_max[pos[0]] + 1)
        #print("step 1")
        #print(tmp[pos[0]][pos[1]]-1)
        # step 2
        vel = tmp[pos[0]][pos[1]] - 1
        lane_index = lanes[pos[0]].index(pos[1])
        if lane_index == len(lanes[pos[0]]) - 1 and vel + pos[1] >= len(tmp):
            d = abs((lanes[pos[0]][0] - lane_index) % (len(tmp) - 1))
        elif i == len(carOrdinates) - 1:
            d = abs(
                (lanes[pos[0]][0] - lane_index) % (len(tmp) - 1)
                + ((len(tmp) - 1) - lane_index)
            )
        else:
            d = lanes[pos[0]][lane_index + 1] - pos[1]
        if abs(d) <= tmp[pos[0]][pos[1]] - 1:
            tmp[pos[0]][pos[1]] = max(abs(d), 1)
        #print("d")
        #print(d)
        #print("step 2")
        #print(tmp[pos[0]][pos[1]]-1)
        # step 3
        if rand.random() <= p_y:
            tmp[pos[0]][pos[1]] = max(tmp[pos[0]][pos[1]] - 1, 1)
        # print("step 3")
        # print(tmp[pos[0]][pos[1]])
        # step 4
    for pos in carOrdinates:
        vel = tmp[pos[0]][pos[1]] - 1
        #print("final")
        print(vel)
        if pos[1] + vel >= len(tmp[pos[0]]):
            road[pos[0]][(pos[1] + vel) % (len(tmp[pos[0]]) - 1)] = tmp[pos[0]][pos[1]]
            if vel != 0:
                road[pos[0]][pos[1]] = 0
        else:
            road[pos[0]][pos[1] + vel] = tmp[pos[0]][pos[1]]
            if vel != 0:
                road[pos[0]][pos[1]] = 0
    t += 1

import random as rand
import numpy as np
from matplotlib import pyplot as plt

# set number of lanes and length of the road
num_lanes = 3
road_length = 7
empty_road = np.array([[0 for i in range(road_length)] for i in range(num_lanes)])

#generates initial configuration of simulation
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


# set maximum speed, per lane, of the road
v_max = [6, 5, 4]
# set total number of cars on road
n = 5
road = populateRoad(n, v_max, empty_road)
# probability of  acceleration noise
p_y = 0.1
# probability of lane change
p_x = 0.05
# record initial road configuration
position = np.nonzero(road)
nonZs = [[]]
for i, x in enumerate(position[0]):
    nonZs[0].append([x, position[1][i]])
# nonZs=np.array(nonZs)

t = 0
# set duration of simulation
finish = 10

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
    for i, pos in enumerate(carOrdinates):
        vel = tmp[pos[0]][pos[1]] - 1
        lane_index = lanes[pos[0]].index(pos[1])
        #calculate distance headway
        if len(lanes[pos[0]]) == 1:
            d = len(tmp[pos[0]])
        elif len(lanes[pos[0]]) != 1 and lane_index == len(lanes[pos[0]]) - 1 and vel + pos[1] >= len(tmp[pos[0]]):
            d = abs((lanes[pos[0]][0] - pos[1]) % (len(tmp[pos[0]]) - 1))
        elif len(lanes[pos[0]]) != 1 and lane_index == len(lanes[pos[0]]) - 1:
            d = abs((lanes[pos[0]][0] - pos[1]) % (len(tmp[pos[0]]) - 1) + ((len(tmp[pos[0]]) - 1) - pos[1]))
        else:
            d = lanes[pos[0]][lane_index + 1] - pos[1]
        #implement lane changing rules and updates
        #calculate forwards and backwards distance headways
        tmp_lanes=lanes
        if pos[0]==0:
            d_r=0
            d_r_b=0
            if tmp[pos[0]+1][pos[1]]==0:
                tmp_lanes[pos[0]+1].append(pos[1])
                tmp_lanes[pos[0]+1].sort()
                lane_index_r = tmp_lanes[pos[0]+1].index(pos[1])
                if len(tmp_lanes[pos[0]+1]) == 1:
                    d_r = len(tmp[pos[0]+1])
                    d_r_b=len(tmp[pos[0]+1])
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == len(tmp_lanes[pos[0]+1]) - 1 and vel + pos[1] >= len(tmp[pos[0]+1]):
                    d_r = abs((tmp_lanes[pos[0]+1][0] - pos[1]) % (len(tmp[pos[0]+1]) - 1))
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == len(tmp_lanes[pos[0]+1]) - 1:
                    d_r = abs((lanes[pos[0]+1][0] - pos[1]) % (len(tmp[pos[0]+1]) - 1) + ((len(tmp) - 1) - pos[1]))
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == 0 and tmp_lanes[pos[0]+1][0] > tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1]:
                    d_r_b=abs(pos[1]-tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1])
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == 0 and tmp_lanes[pos[0]+1][0] < tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1]:
                    d_r_b=abs(pos[1]-tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1]) % (len(tmp[pos[0]+1])-1)
                else:
                    d_r = tmp_lanes[pos[0]+1][lane_index_r + 1] - pos[1]
                    d_r_b=pos[1]-tmp_lanes[pos[0]+1][lane_index_r-1]
                if d_r > d and d_r_b >=v_max[1]:
                    if rand.random() <= p_x:
                        tmp[pos[0]+1][pos[1]]=vel
                        tmp[pos[0]][pos[1]]=0
                        pos[0]+=1
        elif pos[0]==1:
            d_r=0
            d_r_b=0
            d_l=0
            d_l_b=0
            if tmp[pos[0]-1][pos[1]]==0 and tmp[pos[0]+1][pos[1]]!=0:
                tmp_lanes[pos[0]-1].append(pos[1])
                tmp_lanes[pos[0]-1].sort()
                lane_index_l = tmp_lanes[pos[0]-1].index(pos[1])
                if len(tmp_lanes[pos[0]-1]) == 1:
                    d_l = len(tmp[pos[0]-1])
                    d_l_b=len(tmp[pos[0]-1])
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == len(tmp_lanes[pos[0]-1]) - 1 and vel + pos[1] >= len(tmp[pos[0]-1]):
                    d_l = abs((tmp_lanes[pos[0]-1][0] - pos[1]) % (len(tmp[pos[0]-1]) - 1))
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == len(tmp_lanes[pos[0]-1]) - 1:
                    d_l = abs((lanes[pos[0]-1][0] - pos[1]) % (len(tmp[pos[0]-1]) - 1) + ((len(tmp) - 1) - pos[1]))
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == 0 and tmp_lanes[pos[0]-1][0] > tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1]:
                    d_l_b=abs(pos[1]-tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1])
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == 0 and tmp_lanes[pos[0]-1][0] < tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1]:
                    d_l_b=abs(pos[1]-tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1]) % (len(tmp[pos[0]-1])-1)
                else:
                    d_l = tmp_lanes[pos[0]-1][lane_index_l + 1] - pos[1]
                    d_l_b=pos[1]-tmp_lanes[pos[0]-1][lane_index_l-1]
                if d_l > d and d_l_b >=v_max[1]:
                    if rand.random() <= p_x:
                        tmp[pos[0]-1][pos[1]]=vel
                        tmp[pos[0]][pos[1]]=0
                        pos[0]-=1
            elif tmp[pos[0]+1][pos[1]]==0 and tmp[pos[0]-1][pos[1]]!=0:
                tmp_lanes[pos[0]+1].append(pos[1])
                tmp_lanes[pos[0]+1].sort()
                lane_index_r = tmp_lanes[pos[0]+1].index(pos[1])
                if len(tmp_lanes[pos[0]+1]) == 1:
                    d_r = len(tmp[pos[0]+1])
                    d_r_b=len(tmp[pos[0]+1])
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == len(tmp_lanes[pos[0]+1]) - 1 and vel + pos[1] >= len(tmp[pos[0]+1]):
                    d_r = abs((tmp_lanes[pos[0]+1][0] - pos[1]) % (len(tmp[pos[0]+1]) - 1))
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == len(tmp_lanes[pos[0]+1]) - 1:
                    d_r = abs((lanes[pos[0]+1][0] - pos[1]) % (len(tmp[pos[0]+1]) - 1) + ((len(tmp) - 1) - pos[1]))
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == 0 and tmp_lanes[pos[0]+1][0] > tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1]:
                    d_r_b=abs(pos[1]-tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1])
                elif len(tmp_lanes[pos[0]+1]) != 1 and lane_index_r == 0 and tmp_lanes[pos[0]+1][0] < tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1]:
                    d_r_b=abs(pos[1]-tmp_lanes[pos[0]+1][len(tmp_lanes[pos[0]+1])-1]) % (len(tmp[pos[0]+1])-1)
                else:
                    d_r = tmp_lanes[pos[0]+1][lane_index_r + 1] - pos[1]
                    d_r_b=pos[1]-tmp_lanes[pos[0]+1][lane_index_r-1]
                if d_r > d and d_r_b >=v_max[1]:
                    if rand.random() <= p_x:
                        tmp[pos[0]+1][pos[1]]=vel
                        tmp[pos[0]][pos[1]]=0
                        pos[0]+=1
            elif tmp[pos[0]-1][pos[1]]==0 and tmp[pos[0]+1][pos[1]]==0:
                tmp_lanes_r=lanes
                tmp_lanes_r[pos[0]+1].append(pos[1])
                tmp_lanes_r[pos[0]+1].sort()
                lane_index_r = tmp_lanes_r[pos[0]+1].index(pos[1])
                if len(tmp_lanes_r[pos[0]+1]) == 1:
                    d_r = len(tmp[pos[0]+1])
                    d_r_b=len(tmp[pos[0]+1])
                elif len(tmp_lanes_r[pos[0]+1]) != 1 and lane_index_r == len(tmp_lanes_r[pos[0]+1]) - 1 and vel + pos[1] >= len(tmp[pos[0]+1]):
                    d_r = abs((tmp_lanes_r[pos[0]+1][0] - pos[1]) % (len(tmp[pos[0]+1]) - 1))
                elif len(tmp_lanes_r[pos[0]+1]) != 1 and lane_index_r == len(tmp_lanes_r[pos[0]+1]) - 1:
                    d_r = abs((tmp_lanes_r[pos[0]+1][0] - pos[1]) % (len(tmp[pos[0]+1]) - 1) + ((len(tmp) - 1) - pos[1]))
                elif len(tmp_lanes_r[pos[0]+1]) != 1 and lane_index_r == 0 and tmp_lanes_r[pos[0]+1][0] > tmp_lanes_r[pos[0]+1][len(tmp_lanes_r[pos[0]+1])-1]:
                    d_r_b=abs(pos[1]-tmp_lanes_r[pos[0]+1][len(tmp_lanes_r[pos[0]+1])-1])
                elif len(tmp_lanes_r[pos[0]+1]) != 1 and lane_index_r == 0 and tmp_lanes_r[pos[0]+1][0] < tmp_lanes_r[pos[0]+1][len(tmp_lanes_r[pos[0]+1])-1]:
                    d_r_b=abs(pos[1]-tmp_lanes_r[pos[0]+1][len(tmp_lanes_r[pos[0]+1])-1]) % (len(tmp[pos[0]+1])-1)
                else:
                    d_r = tmp_lanes_r[pos[0]+1][lane_index_r + 1] - pos[1]
                    d_r_b=pos[1]-tmp_lanes_r[pos[0]+1][lane_index_r-1]
                tmp_lanes_l=lanes
                tmp_lanes_l[pos[0]-1].append(pos[1])
                tmp_lanes_l[pos[0]-1].sort()
                lane_index_l = tmp_lanes_l[pos[0]-1].index(pos[1])
                if len(tmp_lanes_l[pos[0]-1]) == 1:
                    d_l = len(tmp[pos[0]-1])
                    d_l_b=len(tmp[pos[0]-1])
                elif len(tmp_lanes_l[pos[0]-1]) != 1 and lane_index_l == len(tmp_lanes_l[pos[0]-1]) - 1 and vel + pos[1] >= len(tmp[pos[0]-1]):
                    d_l = abs((tmp_lanes_l[pos[0]-1][0] - pos[1]) % (len(tmp[pos[0]-1]) - 1))
                elif len(tmp_lanes_l[pos[0]-1]) != 1 and lane_index_l == len(tmp_lanes_l[pos[0]-1]) - 1:
                    d_l = abs((tmp_lanes_l[pos[0]-1][0] - pos[1]) % (len(tmp[pos[0]-1]) - 1) + ((len(tmp) - 1) - pos[1]))
                elif len(tmp_lanes_l[pos[0]-1]) != 1 and lane_index_l == 0 and tmp_lanes_l[pos[0]-1][0] > tmp_lanes_l[pos[0]-1][len(tmp_lanes_l[pos[0]-1])-1]:
                    d_l_b=abs(pos[1]-tmp_lanes_l[pos[0]-1][len(tmp_lanes_l[pos[0]-1])-1])
                elif len(tmp_lanes_l[pos[0]-1]) != 1 and lane_index_l == 0 and tmp_lanes_l[pos[0]-1][0] < tmp_lanes_l[pos[0]-1][len(tmp_lanes_l[pos[0]-1])-1]:
                    d_l_b=abs(pos[1]-tmp_lanes_l[pos[0]-1][len(tmp_lanes_l[pos[0]-1])-1]) % (len(tmp[pos[0]-1])-1)
                else:
                    d_l = tmp_lanes_l[pos[0]-1][lane_index_l + 1] - pos[1]
                    d_l_b=pos[1]-tmp_lanes_l[pos[0]-1][lane_index_l-1]
                if d_r > d and d_r > d_l and d_r_b >= v_max[2]:
                    if rand.random() <= p_x:
                        tmp[pos[0]+1][pos[1]]=vel
                        tmp[pos[0]][pos[1]]=0
                        pos[0]+=1
                elif d_l > d and d_l > d_r and d_l_b >= v_max[0]:
                    if rand.random() <= p_x:
                        tmp[pos[0]-1][pos[1]]=vel
                        tmp[pos[0]][pos[1]]=0
                        pos[0]-=1
                elif d_l > d and d_r > d and d_l == d_r:
                    if d_r_b >= v_max[2] and d_l_b < v_max[0]:
                        if rand.random() <= p_x:
                            tmp[pos[0]+1][pos[1]]=vel
                            tmp[pos[0]][pos[1]]=0
                            pos[0]+=1 
                    elif d_r_b < v_max[2] and d_l_b >= v_max[0]:
                        if rand.random() <= p_x:
                            tmp[pos[0]-1][pos[1]]=vel
                            tmp[pos[0]][pos[1]]=0
                            pos[0]-=1
                    elif d_r_b >= v_max[2] and d_l_b >= v_max[0]:
                        if rand.random() <= p_x:
                            tmp[pos[0]-1][pos[1]]=vel
                            tmp[pos[0]][pos[1]]=0
                            pos[0]-=1
                    else:
                        pass
            else:
                pass
        else:
            d_l=0
            d_l_b=0
            if tmp[pos[0]-1][pos[1]]==0:
                tmp_lanes[pos[0]-1].append(pos[1])
                tmp_lanes[pos[0]-1].sort()
                lane_index_l = tmp_lanes[pos[0]-1].index(pos[1])
                if len(tmp_lanes[pos[0]-1]) == 1:
                    d_l = len(tmp[pos[0]-1])
                    d_l_b=len(tmp[pos[0]-1])
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == len(tmp_lanes[pos[0]-1]) - 1 and vel + pos[1] >= len(tmp[pos[0]-1]):
                    d_l = abs((tmp_lanes[pos[0]-1][0] - pos[1]) % (len(tmp[pos[0]-1]) - 1))
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == len(tmp_lanes[pos[0]-1]) - 1:
                    d_l = abs((lanes[pos[0]-1][0] - pos[1]) % (len(tmp[pos[0]-1]) - 1) + ((len(tmp) - 1) - pos[1]))
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == 0 and tmp_lanes[pos[0]-1][0] > tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1]:
                    d_l_b=abs(pos[1]-tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1])
                elif len(tmp_lanes[pos[0]-1]) != 1 and lane_index_l == 0 and tmp_lanes[pos[0]-1][0] < tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1]:
                    d_l_b=abs(pos[1]-tmp_lanes[pos[0]-1][len(tmp_lanes[pos[0]-1])-1]) % (len(tmp[pos[0]-1])-1)
                else:
                    d_l = tmp_lanes[pos[0]-1][lane_index_l + 1] - pos[1]
                    d_l_b=pos[1]-tmp_lanes[pos[0]-1][lane_index_l-1]
                if d_l > d and d_l_b >=v_max[1]:
                    if rand.random() <= p_x:
                        tmp[pos[0]-1][pos[1]]=vel
                        tmp[pos[0]][pos[1]]=0
                        pos[0]-=1
        # implement 1D NaSch algo for each car for longitudinal traversal
        #step 1
        tmp[pos[0]][pos[1]] = min(tmp[pos[0]][pos[1]] + 1, v_max[pos[0]] + 1)
        # step 2
        #!!!need to 
        if abs(d) <= vel:
            tmp[pos[0]][pos[1]] = max(abs(d), 1)
        # step 3
        if rand.random() <= p_y:
            tmp[pos[0]][pos[1]] = max(tmp[pos[0]][pos[1]] - 1, 1)
        # step 4
    for pos in carOrdinates:
        vel = tmp[pos[0]][pos[1]] - 1
        if pos[1] + vel >= len(tmp[pos[0]]):
            road[pos[0]][(pos[1] + vel) % (len(tmp[pos[0]]) - 1)] = tmp[pos[0]][pos[1]]
            if vel != 0:
                road[pos[0]][pos[1]] = 0
        else:
            road[pos[0]][pos[1] + vel] = tmp[pos[0]][pos[1]]
            if vel != 0:
                road[pos[0]][pos[1]] = 0
    print(len(carOrdinates))
    t += 1

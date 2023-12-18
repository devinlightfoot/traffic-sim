import random as rand
import numpy as np
from matplotlib import pyplot as plt

# create single lane, 7.5 km long empty highway
# with v_max=5, each cell corresponds to 7.5 m and each time step is 1 s
zeros = bytearray(1000)
empty_road = np.array([int(str(x)) for x in zeros])


# function that adds cars to empty roar
def populateRoad(n, v_max, empty_road):
    l = len(empty_road)
    populated = empty_road
    i = 1
    while i <= n:
        randCell = rand.randint(0, l - 1)
        if populated[randCell] == 0:
            i += 1
            populated[randCell] = rand.randint(1, v_max + 1)
    return populated


v_max = 5
n = 200
road = populateRoad(n, v_max, empty_road)
# probability of overreaction
p = 0.25
q = 0.1
# array with x(t) for all cars
position = np.nonzero(road)[0]
t = 0
# set duration of simulation
finish = 600

while t < finish:
    tmp = road
    carArr = np.nonzero(road)[0]
    carArr = np.sort(carArr)
    if t > 0:
        position = np.vstack((position, carArr))
    for i, posIndex in enumerate(carArr):
        # implement NaSch algorithm for each car
        # step 1
        # implement BJH slow-to-start rule
        if tmp[posIndex] == 1:
            if rand.random() <= q:
                tmp[posIndex] = 1
            else:
                tmp[posIndex] = min(tmp[posIndex] + 1, v_max + 1)
        else:
            tmp[posIndex] = min(tmp[posIndex] + 1, v_max + 1)
        # step 2
        vel = tmp[posIndex] - 1
        if i == len(carArr) - 1 and vel + posIndex >= len(tmp):
            d = (carArr[0] - (posIndex)) % (len(tmp))
        elif i == len(carArr) - 1:
            d = ((carArr[0] - (posIndex)) % (len(tmp))) + ((len(tmp) - 1) - posIndex)
        else:
            d = carArr[i + 1] - posIndex
        if abs(d) <= tmp[posIndex]:
            tmp[posIndex] = max(abs(d), 1)
        # step 3
        trial = rand.random()
        if trial <= p:
            tmp[posIndex] = max(tmp[posIndex] - 1, 1)
    # step 4
    for posIndex in carArr:
        # implement periodic boundary conditions
        vel = tmp[posIndex] - 1
        if posIndex + vel >= len(tmp):
            road[(posIndex + vel) % (len(tmp))] = tmp[posIndex]
            if vel != 0:
                road[posIndex] = 0
        else:
            road[posIndex + vel] = tmp[posIndex]
            if vel != 0:
                road[posIndex] = 0
    t += 1
# create arrays necessary for plotting

posArr = [[] for i in range(n)]
tArr = [i for i in range(len(position))]
for instance in position:
    for i, car in enumerate(instance):
        posArr[i].append(car)
fig, ax = plt.subplots()
for i, orbit in enumerate(posArr):
    ax.scatter(tArr, orbit, label=str(i), marker=".", color="b", linewidth=0, s=5)
ax.set_xlabel("t (s)")
ax.set_ylabel("x (m)")
plt.show()

from GA import GA
from main import Device
import time
import matplotlib.pyplot as plt
import math

#----------------------test----------------------------
start = time.time()  # 开始计时


device = Device(numOfServer = 8, numOfBank = 3, numOfME = 50, taskSize = [600, 1200],
                taskCycle = [0.8, 1.5], computationPowerOfServer = [18, 28], computationPowerOfME = [0.5, 1],
                coEfficient = [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]])

device.initPosition()
device.proposedAlgorithm()
# print('System cost of all local:{}'.format(device.getTotalCostOfLocal()))
print('System cost of algorithm:{}'.format(device.getTotalCostOfAlgorithm()))
# device.restoreState()
# device.randomDecision()
# print('System cost of random:{}'.format(device.getTotalCostOfAlgorithm()))
# device.restoreState()


# -------------------------test GA algorithm--------------------------
N = 100 # 种群规模
M = 5 # 每一代前Mfitness的个体
x_num = device.numOfME # 决策变量个数
max_gen = 20000 # 最大进化代数
ga = GA(device, N = N, M = M, f_num = 1, x_num = x_num, max_gen = max_gen, pl = 0.1, pr = 0.15, k1 = 0.5, k2 = 0.8, k3 = 0.25, k4 = 0.7)

population = ga.initial() # 初始化种群
max_index = 0
gen = 1 # 迭代代数
while(gen <= max_gen):

    population_cross = ga.cross(population, tournament = 2) # 交叉并进行变异生成 N - M 个后代
    population = ga.copyBest(population, population_cross) # 留下fitness最好的M个基因

    #-----------------------画图--------------------------------
    if(gen % 5000 == 0):
        x, y = [], []
        max = 0
        for i in range(len(population)):
            if(max < population[i].fitness):
                max = population[i].fitness
                max_index = i
            x.append(i)
            y.append(1 / population[i].fitness)
            plt.scatter(x, y, marker = 'o',color = 'red', s = 10)
            plt.xlabel('x')
            plt.ylabel('f(x)')
        plt.show()
        print("{0} gen has completed! --------> Optimium:{1}".format(gen, 1 / max))
    gen = gen + 1


# 从新设置device的decision
device.decision = [[[0] * (device.numOfServer + 1), [0] * (device.numOfBank + 1)] for j in range(device.numOfME + 1)]  # 清空
device.loanerOfBank = [[] for j in range(device.numOfBank + 1)]  # 清空
device.offloaderOfServer = [[] for j in range(device.numOfServer + 1)]  # 清空
for j in range(len(population[max_index].x)):
    random_num = population[max_index].x[j]
    if (random_num == 0):
        pass
    else:
        m = math.floor((random_num - 1) / device.numOfServer) + 1
        s = random_num - (m - 1) * device.numOfServer
        device.decision[j + 1][0][s] = 1
        device.decision[j + 1][1][m] = 1
        device.offloaderOfServer[s].append(j + 1)
        device.loanerOfBank[m].append(j + 1)

device.proposedAlgorithm()
print('Total cost of algorithm:{}'.format(device.getTotalCostOfAlgorithm()))


end = time.time()
print("循环时间：{:.2f}秒".format(end - start))

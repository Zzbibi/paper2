from NSGA2 import NSGA2
from main import Device
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#----------------------test----------------------------
start = time.time()  # 开始计时


device = Device(numOfServer = 4, numOfBank = 3, numOfME = 40, taskSize = [200, 800],
                taskCycle = [0.5, 1.5], computationPowerOfServer = [10, 30], computationPowerOfME = [0.5, 1],
                coEfficient = [0.05, 0.004])
device.initPosition()
device.proposedAlgorithm()
print('System cost of all local:{}'.format(device.getTotalCostOfLocal()))
print('System cost of algorithm:{}'.format(device.getTotalCostOfAlgorithm()))
device.restoreState()
device.randomDecision()
print('System cost of random:{}'.format(device.getTotalCostOfAlgorithm()))
device.restoreState()

N = 60# 种群规模
x_num = device.numOfME # 决策变量个数
max_gen = 5000 # 最大进化代数
nsga = NSGA2(device, N = N, f_num = 1, x_num = x_num, max_gen = max_gen, pc = 0.8, pm = 1 / x_num, yita1 = 2, yita2 = 5)
chromo = nsga.initial() # 初始化种群
F, chromo_non = nsga.non_domination_sort(chromo) # 种群的非支配排序
chromo = nsga.crowding_distance_sort(F, chromo_non) # 种群的拥挤度排序
gen = 1 # 迭代代数
while(gen <= max_gen):
    for i in range(N):
        chromo_parent_1 = nsga.tournament_selection2(chromo)
        chromo_parent_2 = nsga.tournament_selection2(chromo)
        chromo_parent = chromo_parent_1 + chromo_parent_2
        chromo_offspring = nsga.cross_mutation(chromo_parent) # 交叉变异
        # 精英保留策略
        # 将子代和父代合并
        combine_chromo = chromo + chromo_offspring
        nsga.x_num = len(combine_chromo)
        F, combine_chromo1 = nsga.non_domination_sort(combine_chromo)  # 种群的非支配排序
        nsga.x_num = x_num
        combine_chromo2 = nsga.crowding_distance_sort(F, combine_chromo1)  # 种群的拥挤度排序
        chromo = nsga.elitism(combine_chromo2)
    #-----------------------画图--------------------------------
    if(gen % 50 == 0):
        x, y = [], []
        for i in range(len(chromo)):
            x.append(i)
            y.append(chromo[i].f[0])
            plt.scatter(x, y, marker = 'o',color = 'red', s = 10)
            plt.xlabel('x')
            plt.ylabel('f(x)')
        plt.show()
        print("%d gen has completed!\n" % gen)
    gen = gen + 1;
end = time.time()
print("循环时间：{:.2f}秒".format(end - start))

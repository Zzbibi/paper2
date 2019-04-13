from main import Device
from GA import GA
import pandas as pd
import math

round = 50


#-----------------数据CPU的影响-----------------------


dataCycArr = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
len_dataCycArr = len(dataCycArr)

systemcost = [[] for i in range(5)]
all_systemcost = []

for r in range(round):

    for i in range(len_dataCycArr):

        print('----------------------Data Cycle:{}-----------------------'.format(dataCycArr[i]))

        device = Device(numOfServer=8, numOfBank=3, numOfME=50, taskSize=[600, 1200],
                        taskCycle=[dataCycArr[i], dataCycArr[i]], computationPowerOfServer=[18, 28], computationPowerOfME=[0.5, 1],
                        coEfficient=[[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]])

        device.initPosition()

        systemcost[4].append(device.getTotalCostOfLocal())
        print('Total cost of local:{}'.format(device.getTotalCostOfLocal()))

        device.proposedAlgorithm()
        systemcost[0].append(device.getTotalCostOfAlgorithm())
        print('Total cost of algorithm:{}'.format(device.getTotalCostOfAlgorithm()))

        device.restoreState()
        device.randomDecision()
        systemcost[1].append(device.getTotalCostOfAlgorithm())
        print('Total cost of random:{}'.format(device.getTotalCostOfAlgorithm()))

        device.restoreState()
        device.allOffloading()
        systemcost[2].append(device.getTotalCostOfAlgorithm())
        print('Total cost of all offloading:{}'.format(device.getTotalCostOfAlgorithm()))

        device.restoreState()
        N = 150  # 种群规模
        M = 5  # 每一代前Mfitness的个体
        x_num = device.numOfME  # 决策变量个数
        max_gen = 200000  # 最大进化代数
        ga = GA(device, N=N, M=M, f_num=1, x_num=x_num, max_gen=max_gen, pl=0.1, pr=0.2, k1=0.5, k2=0.8, k3=0.2, k4=0.8)
        population = ga.initial()  # 初始化种群
        gen = 1  # 迭代代数
        max_index = 0  # 找到适应度最好的个体
        while (gen <= max_gen):
            population_cross = ga.cross(population, tournament=2)  # 交叉并进行变异生成 N - M 个后代
            population = ga.copyBest(population, population_cross)  # 留下fitness最好的M个基因
            # -----------------------画图--------------------------------
            if (gen % 10000 == 0):
                x, y = [], []
                max = 0
                for j in range(len(population)):
                    if (max < population[j].fitness):
                        max = population[j].fitness
                        max_index = j
                print("{0} gen has completed! --------> Optimium:{1}".format(gen, 1 / max))
            gen = gen + 1

        # 从新设置device的decision
        device.decision = [[[0] * (device.numOfServer + 1), [0] * (device.numOfBank + 1)] for j in
                           range(device.numOfME + 1)]  # 清空
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

        systemcost[3].append(1 / population[max_index].fitness)
        print('Total cost of ga:{}'.format(1 / population[max_index].fitness))

    all_systemcost.append(systemcost)

result = []
for i in range(5):
    temp = []
    for j in range(len_dataCycArr):
        sum = 0
        for k in range(round):
            sum = sum + all_systemcost[k][i][j]
        sum = sum / round
        temp.append(sum)
    result.append(temp)



writer = pd.ExcelWriter(r'numberresult/datacycle.xlsx')
df_systemcost = pd.DataFrame(data = {'Game': result[0], 'Random': result[1], 'alloffloading':result[2], 'Ga':result[3], 'Local':result[4]})
df_systemcost.to_excel(writer, 'sheet2')
writer.save()
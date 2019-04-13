from main import Device
from GA import GA
import pandas as pd
import math

round = 50


#-------------两张图(收益ME)----------------


numOfMEArr = [20, 30, 40, 50, 60, 70, 80, 90]
len_numOfMEArr = len(numOfMEArr)

numOfBeneficalME = [[] for i in range(4)]
ratioOfOfflaoding = [[] for i in range(4)]
systemcost = [[] for i in range(5)]

all_numOfBeneficalME = []
all_ratioOfOfflaoding = []
all_systemcost = []

for r in range(round):

    for i in range(len(numOfMEArr)):

        print('----------------------Number of ME:{}-----------------------'.format(numOfMEArr[i]))

        device = Device(numOfServer=8, numOfBank=3, numOfME=numOfMEArr[i], taskSize=[600, 1200],
                        taskCycle=[0.8, 1.5], computationPowerOfServer=[18, 28], computationPowerOfME=[0.5, 1],
                        coEfficient=[[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]])

        device.initPosition()

        systemcost[4].append(device.getTotalCostOfLocal())
        print('Total cost of local:{}'.format(device.getTotalCostOfLocal()))

        device.proposedAlgorithm()
        numOfBeneficalME[0].append(device.numOfBenificialME())
        ratioOfOfflaoding[0].append(device.numOfOffloading() / device.numOfME)
        systemcost[0].append(device.getTotalCostOfAlgorithm())
        print('Total cost of algorithm:{}'.format(device.getTotalCostOfAlgorithm()))
        print('Number of benificail ME, offloaidng ratio in GAME:{0}, {1}'.format(numOfBeneficalME[0][i], ratioOfOfflaoding[0][i]))

        device.restoreState()
        device.randomDecision()
        numOfBeneficalME[1].append(device.numOfBenificialME())
        ratioOfOfflaoding[1].append(device.numOfOffloading() / device.numOfME)
        systemcost[1].append(device.getTotalCostOfAlgorithm())
        print('Total cost of random:{}'.format(device.getTotalCostOfAlgorithm()))
        print('Number of benificail ME, offloaidng ratio in RANDOM:{0}, {1}'.format(numOfBeneficalME[1][i], ratioOfOfflaoding[1][i]))

        device.restoreState()
        device.allOffloading()
        numOfBeneficalME[2].append(device.numOfBenificialME())
        ratioOfOfflaoding[2].append(device.numOfOffloading() / device.numOfME)
        systemcost[2].append(device.getTotalCostOfAlgorithm())

        # N = 100  # 种群规模
        # M = 10  # 每一代前Mfitness的个体
        # x_num = device.numOfME  # 决策变量个数
        # max_gen = 100000  # 最大进化代数
        # ga = GA(device, N=N, M=M, f_num=1, x_num=x_num, max_gen=max_gen, pm=0.6, pl=0.1, pr=0.2)
        # population = ga.initial()  # 初始化种群
        # gen = 1  # 迭代代数
        # max_index = 0  # 找到适应度最好的个体
        # while (gen <= max_gen):
        #     population_cross = ga.cross(population, tournament=2)  # 交叉并进行变异生成 N - M 个后代
        #     population = ga.copyBest(population, population_cross)  # 留下fitness最好的M个基因
        #     # -----------------------画图--------------------------------
        #     if (gen % 10000 == 0):
        #         x, y = [], []
        #         max = 0
        #         for j in range(len(population)):
        #             if (max < population[j].fitness):
        #                 max = population[j].fitness
        #                 max_index = j
        #         print("{0} gen has completed! --------> Optimium:{1}".format(gen, 1 / max))
        #     gen = gen + 1
        #
        # # 从新设置device的decision
        # device.decision = [[[0] * (device.numOfServer + 1), [0] * (device.numOfBank + 1)] for j in range(device.numOfME + 1)]  # 清空
        # device.loanerOfBank = [[] for j in range(device.numOfBank + 1)]  # 清空
        # device.offloaderOfServer = [[] for j in range(device.numOfServer + 1)]  # 清空
        # for j in range(len(population[max_index].x)):
        #     random_num = population[max_index].x[j]
        #     if (random_num == 0):
        #         pass
        #     else:
        #         m = math.floor((random_num - 1) / device.numOfServer) + 1
        #         s = random_num - (m - 1) * device.numOfServer
        #         device.decision[j + 1][0][s] = 1
        #         device.decision[j + 1][1][m] = 1
        #         device.offloaderOfServer[s].append(j + 1)
        #         device.loanerOfBank[m].append(j + 1)
        #
        # numOfBeneficalME[2].append(device.numOfBenificialME())
        # ratioOfOfflaoding[2].append(device.numOfOffloading() / device.numOfME)
        # systemcost[2].append(1 / population[max_index].fitness)
        # print('Total cost of ga:{}'.format(1 / population[max_index].fitness))
        # print('Number of benificail ME, offloaidng ratio in GA:{0}, {1}'.format(numOfBeneficalME[2][i], ratioOfOfflaoding[2][i]))

        device.restoreState()
        N = 150  # 种群规模
        M = 5  # 每一代前Mfitness的个体
        x_num = device.numOfME  # 决策变量个数
        max_gen = 200000  # 最大进化代数
        ga = GA(device, N=N, M=M, f_num=1, x_num=x_num, max_gen=max_gen, pl=0.1, pr=0.2, k1=0.5, k2=0.8, k3=0.3, k4=0.7)
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

        numOfBeneficalME[3].append(device.numOfBenificialME())
        ratioOfOfflaoding[3].append(device.numOfOffloading() / device.numOfME)
        systemcost[3].append(1 / population[max_index].fitness)
        print('Total cost of ga:{}'.format(1 / population[max_index].fitness))
        print('Number of benificail ME, offloaidng ratio in GA:{0}, {1}'.format(numOfBeneficalME[3][i], ratioOfOfflaoding[3][i]))

    all_numOfBeneficalME.append(numOfBeneficalME)
    all_ratioOfOfflaoding.append(ratioOfOfflaoding)
    all_systemcost.append(systemcost)


result_beneficial = []
for i in range(4):
    temp = []
    for j in range(len_numOfMEArr):
        sum = 0
        for k in range(round):
            sum = sum + all_numOfBeneficalME[k][i][j]
        sum = sum / round
        temp.append(sum)
    result_beneficial.append(temp)

result_ratio = []
for i in range(4):
    temp = []
    for j in range(len_numOfMEArr):
        sum = 0
        for k in range(round):
            sum = sum + all_ratioOfOfflaoding[k][i][j]
        sum = sum / round
        temp.append(sum)
    result_ratio.append(temp)

result_systemcost = []
for i in range(5):
    temp = []
    for j in range(len_numOfMEArr):
        sum = 0
        for k in range(round):
            sum = sum + all_systemcost[k][i][j]
        sum = sum / round
        temp.append(sum)
    result_systemcost.append(temp)



writer = pd.ExcelWriter(r'numberresult/beneficialME_offloadingratio.xlsx')
df_benificial = pd.DataFrame(data = {'Game': result_beneficial[0], 'Random': result_beneficial[1], 'alloffloading':result_beneficial[2], 'Ga':result_beneficial[3]})
df_benificial.to_excel(writer, 'sheet1')
df_ratio = pd.DataFrame(data = {'Game': result_ratio[0], 'Random': result_ratio[1], 'alloffloading':result_ratio[2], 'Ga':result_ratio[3]})
df_ratio.to_excel(writer, 'sheet2')
writer.save()

writer1 = pd.ExcelWriter(r'numberresult/numofME.xlsx')
df_systemcost = pd.DataFrame(data = {'Game': result_systemcost[0], 'Random': result_systemcost[1], 'alloffloading':result_systemcost[2], 'Ga': result_systemcost[3], 'local': result_systemcost[4]})
df_systemcost.to_excel(writer1, 'sheet1')
writer1.save()
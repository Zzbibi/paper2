import numpy as np
import math
import random
import time


# 种群中的每一个个体
class Individual():

    def __init__(self, x, device):
        self.x = x # 个体的染色体（一个可行解）
        self.fitness = 0 # 个体的适应度

        # 计算每一个个体的fitness
        numOfServer = device.numOfServer
        numOfBank = device.numOfBank
        numOfME = device.numOfME
        device.decision = [[[0] * (numOfServer + 1), [0] * (numOfBank + 1)] for i in range(numOfME + 1)]  # 清空
        device.loanerOfBank = [[] for i in range(numOfBank + 1)]  # 清空
        device.offloaderOfServer = [[] for i in range(numOfServer + 1)]  # 清空
        for i in range(len(x)):
            random_num = x[i]
            if(random_num == 0):
                pass
            else:
                m = math.floor((random_num - 1) / numOfServer) + 1
                s = random_num - (m - 1) * numOfServer
                device.decision[i + 1][0][s] = 1
                device.decision[i + 1][1][m] = 1
                device.offloaderOfServer[s].append(i + 1)
                device.loanerOfBank[m].append(i + 1)

        f = device.getTotalCostOfAlgorithm()
        self.fitness = 1 / f   # 通过取倒数将最小化目标（非负）转换为最大化目标


class GA(object):

    def __init__(self, device, N, M, f_num, x_num, max_gen, pl, pr, k1, k2, k3, k4):
        self.f_num = f_num # 目标函数个数
        self.N = N # 种群规模
        self.M = M # 留下的种群
        self.x_num = x_num # 决策变量个数
        self.max_gen = max_gen # 最大进化代数
        #self.pm = pm # 变异概率（随机产生一个数，小于此值将进行变异）
        self.pr = pr # 最大将多少比例的基因进行变异
        self.pl = pl # 最少将多少比例的基因进行变异
        self.device = device  # 接收初始化之后的Device对象
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4


    def initial(self, ):
        '''初始化种群'''

        P = [] # 种群
        for i in range(self.N): # 生成N个个体组成一个种群
            chromosome = []  # 染色体(表示一个可行解)
            for j in range(self.x_num):
                numOfServer = self.device.numOfServer
                numOfBank = self.device.numOfBank
                random_num = random.randint(0, numOfServer * numOfBank)
                app = random_num  # 表示每一个决策变量
                chromosome.append(app)
            P.append(Individual(chromosome, self.device))
        return P


    def cross(self, population, tournament = 2):
        '''对种群population进行基因交叉'''

        probability = [0] * self.N # 概率区间
        sum_fitness = 0 # 所有个体的适应度之和
        for i in range(self.N):
            sum_fitness += population[i].fitness
        section = 0
        for i in range(self.N - 1):
            section += population[i].fitness / sum_fitness
            probability[i] = section
        probability[self.N - 1] = 1

        population_off = [] # 保存产生的N - M后代
        for k in range(self.N - self.M):
            objective_index = [0, 0]
            while(objective_index[0] == objective_index[1]): # 找到两个不同的parent
                objective_index = [0, 0] # 再次循环时，重新选择
                for i in range(tournament):
                    k = 0
                    random_num = random.random() # 产生一个0-1之间的随机数
                    for j in range(self.N):
                        if random_num <= probability[j]:
                            objective_index[k] = j
                            k = k + 1
                            break
                        else:
                            pass

            parent1 = population[objective_index[0]]
            parent2 = population[objective_index[1]]
            f_max = self.maxFitness(population)
            f_avg = self.avgFitness(population)
            insert_index = int(round(self.x_num * 0.8)) # 产生插入点，前半段复制fitness大的parent的基因
            big_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
            small_parent = parent1 if parent1.fitness <parent2.fitness else parent2
            offspring = [] # 后代
            for i in range(self.x_num):
                if insert_index > i:
                    offspring.append(big_parent.x[i])
                else:
                    offspring.append(small_parent.x[i])
            offspring_mutation = self.mutation(offspring,f_max, f_avg) # 对基因进行变异
            population_off.append(Individual(offspring_mutation, self.device))
        return population_off


    def mutation(self, chromosome, f_max, f_avg):
        '''对交叉产生的染色体进行变异'''

        f = Individual(chromosome, self.device).fitness
        if (f >= f_avg):
            pm = self.k3 * (f_max - f) / (f_max - f_avg)
        else:
            pm = self.k4

        random_num = random.random()
        if(random_num < pm):
            chromosome_index = []
            for i in range(self.x_num):
                chromosome_index.append(i)
            mutation_num = random.randint(int(round(self.pl * self.x_num)), int(round(self.pr * self.x_num)))
            mutation_index = random.sample(chromosome_index, mutation_num) # 需要变异的基因
            numOfServer = self.device.numOfServer
            numOfBank = self.device.numOfBank
            for i in range(mutation_num):
                app = random.randint(0, numOfServer * numOfBank)
                chromosome[mutation_index[i]] = app
        return chromosome


    def copyBest(self, population_parent, population_off):
        '''从父代中copy出fitness最优的前M个个体'''

        population_sorted = sorted(population_parent, key = lambda individual: individual.fitness, reverse = True) # 根据适应度进行排序
        for i in range(self.M):
            population_off.append(population_sorted[i])
        return population_off

    def maxFitness(self, population):
        '''计算最大适应度'''

        max = 0
        for i in range(len(population)):
            if(max < population[i].fitness):
                max = population[i].fitness
            else:
                pass
        return max

    def avgFitness(self, population):
        '''计算平均适应度'''

        sum = 0
        for i in range(self.N):
            sum += population[i].fitness
        return sum / self.N




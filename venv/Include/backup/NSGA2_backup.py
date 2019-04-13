import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D


class Individual():  # 定目标函数
    def __init__(self, x, device):
        self.x = x # 一条染色体（一个可行解）
        self.nnd = 0
        self.paretorank = 0
        for i in range(len(x)):
            s = x[i][0]
            m = x[i][1]
            if(s == 0):
                pass
            else:
                device.decision[i + 1][0][s] = 1
                device.decision[i + 1][1][m] = 1
                device.offloaderOfServer[s].append(i + 1)
                device.loanerOfBank[m].append(i + 1)
        f = device.getTotalCostOfAlgorithm()
        self.f = [f]


class NSGA2(object):

    def __init__(self, device, N, f_num, x_num, max_gen, pc):
        self.f_num = f_num # 目标函数个数
        self.device = device # 接收初始化之后的Device对象
        self.N = N # 种群规模
        self.x_num = x_num # 决策变量个数
        self.max_gen = max_gen # 最大进化代数
        self.pc = pc # 交叉概率


    def initial(self):
        P = []
        # 种群初始化以及产生lamda
        for i in range(self.N):
            chromo = []  # 染色体(表示一个可行解)
            for j in range(self.x_num):
                app = [0, 0] # 表示每一个决策变量
                flag = random.randint(0, 1)
                if(flag == 0):
                    pass
                else:
                    numOfServer = self.device.numOfServer
                    numOfBank = self.device.numOfBank
                    s = random.randint(1, numOfServer)
                    m = random.randint(1, numOfBank)
                    app[0] = s
                    app[1] = m
                chromo.append(app)
            P.append(Individual(chromo, self.device))
        return P


    def non_domination_sort(self, population):
        # non_domination_sort 初始种群的非支配排序和计算拥挤度
        # 初始化pareto等级为1
        pareto_rank = 1
        F = {}  # 初始化一个字典，用来保存每一个等级的解的集合
        F[pareto_rank] = []  # pareto等级为pareto_rank的集合
        pn = {}
        ps = {}
        for i in range(self.N):
            # 计算出种群中每个个体p的被支配个数n和该个体支配的解的集合s
            pn[i] = 0  # 被支配个体数目n
            ps[i] = []  # 支配解的集合s
            for j in range(self.N):
                less = 0  # y'的目标函数值小于个体的目标函数值数目
                equal = 0  # y'的目标函数值等于个体的目标函数值数目
                greater = 0  # y'的目标函数值大于个体的目标函数值数目
                for k in range(self.f_num):
                    if (population[i].f[k] < population[j].f[k]):
                        less = less + 1
                    elif (population[i].f[k] == population[j].f[k]):
                        equal = equal + 1
                    else:
                        greater = greater + 1
                if (less == 0 and equal != self.f_num):  # 添加条件equal != self.f_num的原因是也算上了自己的目标函数
                    pn[i] = pn[i] + 1
                elif (greater == 0 and equal != self.f_num):
                    ps[i].append(j)
            if (pn[i] == 0):  # 不被任何个体所支配，pareto等级为1
                population[i].paretorank = 1
                F[pareto_rank].append(i) # pareto等级为1的解已经遍历完成
        # 求pareto等级为pareto_rank + 1的个体
        while (len(F[pareto_rank]) != 0):
            temp = []  # 用来保存每一个pareto等级的解的集合
            for i in range(len(F[pareto_rank])):  # pareto等级为pareto_rank的解
                if (len(ps[F[pareto_rank][i]]) != 0):  # 支配解的集合不为空
                    for j in range(len(ps[F[pareto_rank][i]])):
                        pn[ps[F[pareto_rank][i]][j]] = pn[ps[F[pareto_rank][i]][j]] - 1  # nl = nl - 1 被支配个体数-1
                        if pn[ps[F[pareto_rank][i]][j]] == 0:  # 被支配个体数为0，设置其pareto等级
                            population[ps[F[pareto_rank][i]][j]].paretorank = pareto_rank + 1  # 储存个体的等级信息
                            temp.append(ps[F[pareto_rank][i]][j])
            pareto_rank = pareto_rank + 1  # pareto等级+1
            F[pareto_rank] = temp
        return F, population  # 返回分好pareto等级的解的集合，和


    def crowding_distance_sort(self, F, population):
        # 计算拥挤度
        ppp = []
        # 按照pareto等级对种群中的个体进行排序
        temp = sorted(population, key=lambda Individual: Individual.paretorank)  # 按照pareto等级排序后种群
        index1 = []  # index中保存排序之后的种群在未排序种群population中对应的index
        for i in range(len(temp)):
            index1.append(population.index(temp[i]))
            # 对于每个等级的个体开始计算拥挤度
        current_index = 0
        for pareto_rank in range(len(F) - 1):  # 计算F的循环时多了一次空，所以减掉,由于pareto从1开始，再减一次
            nd = np.zeros(len(F[pareto_rank + 1]))  # 拥挤度初始化为0
            y = []  # 储存当前处理的等级的个体
            yF = np.zeros((len(F[pareto_rank + 1]), self.f_num))
            for i in range(len(F[pareto_rank + 1])):
                y.append(temp[current_index + i])
            current_index = current_index + i + 1  # 由于temp是一个list,用current_index来区分pareto_rank
            # 对于每一个目标函数fm
            for i in range(self.f_num):
                # 根据该目标函数值对该等级的个体进行排序
                index_objective = []  # 通过目标函数排序后的个体索引
                objective_sort = sorted(y, key=lambda Individual: Individual.f[i])  # 根据目标函数i的值排序后的个体
                for j in range(len(objective_sort)):  # 按照目标函数值i排序后的个体在y中index
                    index_objective.append(y.index(objective_sort[j]))
                # 记fmax为最大值，fmin为最小值
                fmin = objective_sort[0].f[i]
                fmax = objective_sort[len(objective_sort) - 1].f[i]
                # 对排序后的两个边界拥挤度设为1d和nd设为无穷（边界）
                yF[index_objective[0]][i] = float("inf")
                yF[index_objective[len(index_objective) - 1]][i] = float("inf")
                # 计算nd = nd + (fm(i + 1) - fm(i - 1)) / (fmax - fmin)
                j = 1
                while (j <= (len(index_objective) - 2)):
                    pre_f = objective_sort[j - 1].f[i]
                    next_f = objective_sort[j + 1].f[i]
                    if (fmax - fmin == 0):
                        yF[index_objective[j]][i] = float("inf")
                    else:
                        yF[index_objective[j]][i] = float((next_f - pre_f) / (fmax - fmin))
                    j = j + 1
            # 多个目标函数拥挤度求和
            nd = np.sum(yF, axis=1)
            for i in range(len(y)):
                y[i].nnd = nd[i]
                ppp.append(y[i])
        return ppp  # 算完拥挤度的种群


    def tournament_selection2(self, population):
        # 竞标赛选择
        touranment = 2
        a = round(self.N / 2)
        chromo_candidate = np.zeros(touranment)
        chromo_rank = np.zeros(touranment)
        chromo_distance = np.zeros(touranment)
        chromo_parent = []
        # 选择a个
        for i in range(a):
            for j in range(touranment):
                chromo_candidate[j] = round(self.N * random.random())  # 随机选择一个候选
                if chromo_candidate[j] == self.N:  # 索引不能为N
                    chromo_candidate[j] = self.N - 1
            while (chromo_candidate[0] == chromo_candidate[1]):
                chromo_candidate[0] = round(self.N * random.random())
                if chromo_candidate[0] == self.N:
                    chromo_candidate[0] = self.N - 1
            chromo_rank[0] = population[int(chromo_candidate[0])].paretorank
            chromo_rank[1] = population[int(chromo_candidate[1])].paretorank
            chromo_distance[0] = population[int(chromo_candidate[0])].nnd
            chromo_distance[1] = population[int(chromo_candidate[1])].nnd
            # 取出低等级的个体索引
            minchromo_candidate = np.argmin(chromo_rank)
            # 多个索引按拥挤度排序
            if (chromo_rank[0] == chromo_rank[1]):
                maxchromo_candidate = np.argmax(chromo_distance)
                chromo_parent.append(population[maxchromo_candidate])
            else:
                chromo_parent.append(population[minchromo_candidate])
        return chromo_parent


    def cross_mutation(self, chromo_parent):
        # 模拟二进制交叉和多项式变异
        ###模拟二进制交叉
        chromo_offspring = []
        # 随机选取两个父代个体
        for i in range(round(len(chromo_parent) / 2)):
            parent_1 = round(len(chromo_parent) * random.random())
            if (parent_1 == len(chromo_parent)):
                parent_1 = len(chromo_parent) - 1
            parent_2 = round(len(chromo_parent) * random.random())
            if (parent_2 == len(chromo_parent)):
                parent_2 = len(chromo_parent) - 1
            while (parent_1 == parent_2):
                parent_1 = round(len(chromo_parent) * random.random())
                if (parent_1 == len(chromo_parent)):
                    parent_1 = len(chromo_parent) - 1
            parent1 = chromo_parent[parent_1]
            parent2 = chromo_parent[parent_2]
            off1 = parent1
            off2 = parent2
            # 交叉变异（不知是否合适）
            for i in range(round(self.x_num * self.pc), self.x_num):
                off1.x[i] = parent2.x[i]
                off2.x[i] = parent1.x[i]
            off1 = Individual(off1.x, self.device)
            off2 = Individual(off2.x, self.device)
            chromo_offspring.append(off1)
            chromo_offspring.append(off2)
        return chromo_offspring


    def elitism(self, combine_chromo2):
        # 精英保留策略
        chromo = []
        index1 = 0
        index2 = 0
        # 根据pareto等级从高到低进行排序
        chromo_rank = sorted(combine_chromo2, key=lambda Individual: Individual.paretorank)
        flag = chromo_rank[self.N - 1].paretorank  # 最大pareto等级
        for i in range(len(chromo_rank)):
            if (chromo_rank[i].paretorank == (flag)):  # chromo_rank中pareto等级最大元素的下标
                index1 = i
                break
            else:
                chromo.append(chromo_rank[i])
        for i in range(len(chromo_rank)):
            if (chromo_rank[i].paretorank == (flag + 1)):  #
                index2 = i
                break
        temp = []
        aaa = index1
        if (index2 == 0):
            index2 = len(chromo_rank)
        while (aaa < index2):
            temp.append(chromo_rank[aaa])
            aaa = aaa + 1
        temp_crowd = sorted(temp, key=lambda Individual: Individual.paretorank, reverse=True)
        remainN = self.N - index1
        for i in range(remainN):
            chromo.append(temp_crowd[i])
        return chromo



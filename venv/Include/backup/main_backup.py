import random as rd
import math as math

class Device(object):

    REGION = 1000 # （m）

    BANDWIDTH = 5 * 1000000 # （MHz）

    DTP = 100 # （mWatts） cellular传输功率

    SP = 0.4 # （W） 发送数据的功率

    LAMBDA_T = 0.4

    LAMBDA_E = 0.6

    RHO  = 0.2


    def __init__(self, numOfServer, numOfBank, numOfME, taskSize, taskCycle, computationPowerOfServer,
                 computationPowerOfME, coEfficient):
        '''初始化参数'''

        self.numOfServer = numOfServer  # edge server的数量 编号：1-numOfServer

        self.computationPowerOfServer = [0] * (numOfServer + 1)  # 每个服务器的计算能力
        for i in range(1, numOfServer + 1):
            self.computationPowerOfServer[i] = rd.uniform(computationPowerOfServer[0], computationPowerOfServer[1])

        self.positionOfServer = [[] for i in range(numOfServer + 1)] #每个server的位置

        self.offloaderOfServer = [[] for i in range(numOfServer + 1)]  # 每个server上的offloader

        self.numOfBank = numOfBank # bank的数量 编号：1-numOfBank

        self.loanerOfBank = [[] for i in range(numOfBank + 1)] # 每个bank的loaner

        self.interestRateOfBank = [0] * (numOfBank + 1) # 每个bank的利率

        self.numOfME = numOfME # 设备的数量  编号：1-numOfME

        self.computationPowerOfME = [0] * (numOfME + 1) # 每个ME的计算能力

        for i in range(1, numOfME + 1):
            self.computationPowerOfME[i] = rd.uniform(computationPowerOfME[0], computationPowerOfME[1])

        self.task = [[] for i in range(numOfME + 1)]  # 每个ME的任务

        for i in range(1, numOfME + 1):
            self.task[i].append(rd.uniform(taskSize[0], taskSize[1])) # 任务的数据量
            self.task[i].append(rd.uniform(taskCycle[0], taskCycle[1])) # 任务所需的CPU时钟周期数

        # self.asset = [0] * (numOfME + 1) # 每个ME自身的资产 （暂不考虑）

        #for i in range(1, numOfME + 1):
        #   self.asset = rd.uniform(asset[0], asset[1])

        self.positionOfME = [[] for i in range(numOfME +  1)] # 每个ME的位置

        self.decision = [[[0] * (self.numOfServer + 1), [0] * (self.numOfBank + 1)] for i in range(numOfME + 1)] # 每个ME的决策

        self.coEfficient = coEfficient

        self.distance = [[] for i in range(numOfME + 1)] # ME和server之间的距离

        self.linkRate = [[] for i in range(numOfME + 1)] # ME和server之间的链路速率

        self.intertionTime = 0

        self.potentialEachIteration = []

        self.systemCostEachIteration = []

        self.meCostEachIteration = [[] for i in range(numOfME + 1)]


    def restoreState(self):
        '''恢复到初始状态'''

        for i in range(1, self.numOfServer + 1):
            self.offloaderOfServer[i].clear()
        for i in range(1, self.numOfBank + 1):
            self.loanerOfBank[i].clear()
        for i in range(1, self.numOfBank + 1):
            self.interestRateOfBank[i] = 0
        for i in range(1, self.numOfME + 1):
            for s in range(1, self.numOfServer + 1):
                self.decision[i][0][s] = 0
            for m in range(1, self.numOfBank + 1):
                self.decision[i][1][m] = 0
        self.intertionTime = 0
        self.potentialEachIteration.clear()
        self.systemCostEachIteration.clear()
        for i in range(i, self.numOfME + 1):
            self.meCostEachIteration[i].clear()


    def getInterestRateOfBank(self):
        for i in range(1, self.numOfBank + 1):
            num_loaner = len(self.loanerOfBank[i])
            self.interestRateOfBank[i] = self.coEfficient[0] / self.numOfME + self.coEfficient[1] * num_loaner

    def numOfBenificialME(self):
        '''计算beneficial ME的数量'''

        num_benificialme = 0
        for i in range(1, self.numOfME):
            if (sum(self.decision[i][0]) == 0):  # decision为本地计算
                pass
            else:
                s = self.decision[i][0].index(1)
                m = self.decision[i][1].index(1)
                # 先将i从previousServer, preciousBank中去掉
                self.offloaderOfServer[s].remove(i)
                self.loanerOfBank[m].remove(i)
                if(self.costOfOffloading(i, s, m) < self.costOfLocal(i)):
                    num_benificialme += 1
                # 将i加入previousServer, preciousBank中
                self.offloaderOfServer[s].append(i)
                self.loanerOfBank[m].append(i)
        return num_benificialme

    def numOfOffloading(self):
        '''计算有多少的ME进行了迁移'''

        num_offloading = 0
        for i in range(1, self.numOfME):
            if (sum(self.decision[i][0]) == 0):  # decision为本地计算
                pass
            else:
                num_offloading += 1
        return num_offloading


    def costOfLocal(self, i):
        '''计算ME i的本地消耗'''

        t = self.task[i][1] / self.computationPowerOfME[i]
        e = self.task[i][1] * math.pow(10, 9) * (self.computationPowerOfME[i] ** 2)  * math.pow(10, -9)
        cost = Device.LAMBDA_T * t + Device.LAMBDA_E * e
        # self.localCost.append(cost)
        return cost


    def getTotalCostOfLocal(self):
        '''所有ME都本地计算的总消耗'''

        sum_local = 0
        for i in range(1, self.numOfME + 1):
            sum_local += self.costOfLocal(i)
        return sum_local


    def costOfOffloading(self, i, s, m):
        '''offloading到edge server s, loan从bank m时, ME i的消耗'''
        '''注意num_offloaderOfServer + 1，num_loanerOfBank + 1'''

        t_offload = self.task[i][0] / self.linkRate[i][s]
        e_offload = t_offload * Device.SP
        num_offloaderOfServer = len(self.offloaderOfServer[s]) # server s中有多少个offloader
        t_compute = (self.task[i][1] / self.computationPowerOfServer[s]) * (num_offloaderOfServer + 1)
        num_loanerOfBank = len(self.loanerOfBank[m])  # bank m中有多少loaner
        debt = (1 + self.coEfficient[0] / self.numOfME + self.coEfficient[1] * (num_loanerOfBank + 1)) * (self.task[i][1] * Device.RHO)
        return Device.LAMBDA_T * (t_offload + t_compute) + Device.LAMBDA_E * e_offload + debt


    def getTotalCostOfAlgorithm(self):
        '''所有ME的总消耗'''

        sum_algorithm = 0
        for i in range(1, self.numOfME + 1):
            if(sum(self.decision[i][0]) == 0): # decision为本地计算
                sum_algorithm += self.costOfLocal(i)
            else:
                s = self.decision[i][0].index(1)
                m = self.decision[i][1].index(1)
                # 先将i从previousServer, preciousBank中去掉
                self.offloaderOfServer[s].remove(i)
                self.loanerOfBank[m].remove(i)
                sum_algorithm += self.costOfOffloading(i, s, m)
                # 将i加入previousServer, preciousBank中
                self.offloaderOfServer[s].append(i)
                self.loanerOfBank[m].append(i)
        return sum_algorithm


    def getPotentialFucntion(self):
        firstTerm = 0
        for i in range(1, self.numOfServer + 1):
            for j in range(1, len(self.offloaderOfServer[i]) + 1):
                firstTerm += j / self.computationPowerOfServer[i] * Device.LAMBDA_T
        secondTerm = 0
        for i in range(1, self.numOfBank + 1):
            for j in range(1, len(self.loanerOfBank[i]) + 1):
                secondTerm += self.coEfficient[1] * Device.RHO * j
        thirdTerm = 0
        for i in range(1, self.numOfME + 1):
            if(sum(self.decision[i][0]) == 0):
                continue
            else:
                s = self.decision[i][0].index(1)
                t_offload = self.task[i][0] / self.linkRate[i][s]
                e_offload = t_offload * Device.SP
                thirdTerm += (Device.LAMBDA_E * e_offload + Device.LAMBDA_T * t_offload) / self.task[i][1]
        forthTerm = 0
        for i in range(1, self.numOfME + 1):
            if(sum(self.decision[i][0]) == 0):
                forthTerm += (self.costOfLocal(i) - (1 + self.coEfficient[0] / self.numOfME) * (self.task[i][1] * Device.RHO)) / self.task[i][1]
        #返回potential function的值
        return firstTerm + secondTerm + thirdTerm + forthTerm


    def goToNE(self, bank = True):
        '''达到NE返回True, 未达到返回False'''

        updateAll = []
        update = []
        for i in range(1, self.numOfME + 1):

            # 没有bank的计算迁移（暂不考虑）

            if(sum(self.decision[i][0]) == 0): # 上一步的decision为本地计算
                for s in range(1, self.numOfServer + 1):
                    for m in range(1, self.numOfBank + 1):
                        if(self.costOfOffloading(i, s, m) < self.costOfLocal(i)): # 有更好的决策
                            temp = []
                            temp.append(i)
                            temp.append((s, m))
                            temp.append(self.costOfLocal(i) - self.costOfOffloading(i, s, m))
                            updateAll.append(temp)
                        else:
                            pass
            else:
                previousServer = self.decision[i][0].index(1)
                previousBank = self.decision[i][1].index(1)
                # 先将i从previousServer, preciousBank中去掉
                self.offloaderOfServer[previousServer].remove(i)
                self.loanerOfBank[previousBank].remove(i)
                if(self.costOfLocal(i) < self.costOfOffloading(i, previousServer, previousBank)):
                    temp = []
                    temp.append(i)
                    temp.append((0, 0))
                    temp.append(self.costOfOffloading(i, previousServer, previousBank) - self.costOfLocal(i))
                    updateAll.append(temp)
                else:
                    for s in range(1, self.numOfServer + 1):
                        for m in range(1, self.numOfBank + 1):
                            if(self.costOfOffloading(i, s, m) < self.costOfOffloading(i, previousServer, previousBank)):
                                temp = []
                                temp.append(i)
                                temp.append((s, m))
                                temp.append(self.costOfOffloading(i, previousServer, previousBank) - self.costOfOffloading(i, s, m))
                                updateAll.append(temp)
                            else:
                                pass
                # 将i添加到previousServer, preciousBank中
                self.offloaderOfServer[previousServer].append(i)
                self.loanerOfBank[previousBank].append(i)

        # 如果没有需要更新的，NE已经达到
        if(len(updateAll) == 0):
            return True
        else:
            # 根据cost进行排序
            updateAll.sort(key = lambda x : x[2], reverse = True)
            # 选择使cost减小最多的更新（也即使potential function减小最多的）
            update = updateAll[0]

        # 更新的update中的ME的decision
        updateME = update[0] # update中的ME
        previousServer, previousBank = 0, 0
        currentServer = update[1][0]
        currentBank = update[1][1]
        case1, case2, case3, case4, case5 = False, False, False, False, False
        if(sum(self.decision[updateME][0]) == 0): # previous decision 为local computing
            case1 = True
        else: # previous decision 为computation offloading
            previousServer = self.decision[updateME][0].index(1)
            previousBank = self.decision[updateME][1].index(1)
            if(currentServer == 0 and currentBank == 0):
                case5 = True
            else:
                if (previousServer != currentServer and previousBank == currentBank):
                    case2 = True
                elif(previousServer == currentServer and previousBank != currentBank):
                    case3  = True
                elif(previousServer != currentServer and previousBank != currentBank):
                    case4 = True
        # 更新之前获取potential function的值
        potentialBeforeUpdate = self.getPotentialFucntion()
        if(case1):
            self.offloaderOfServer[currentServer].append(updateME)
            self.loanerOfBank[currentBank].append(updateME)
            self.decision[updateME][0][currentServer] = 1
            self.decision[updateME][1][currentBank] = 1
        if(case2):
            self.offloaderOfServer[previousServer].remove(updateME)
            self.offloaderOfServer[currentServer].append(updateME)
            self.decision[updateME][0][previousServer] = 0
            self.decision[updateME][0][currentServer] = 1
        if(case3):
            self.loanerOfBank[previousBank].remove(updateME)
            self.loanerOfBank[currentBank].append(updateME)
            self.decision[updateME][1][previousBank] = 0
            self.decision[updateME][1][currentBank] = 1
        if(case4):
            self.offloaderOfServer[previousServer].remove(updateME)
            self.offloaderOfServer[currentServer].append(updateME)
            self.loanerOfBank[previousBank].remove(updateME)
            self.loanerOfBank[currentBank].append(updateME)
            self.decision[updateME][0][previousServer] = 0
            self.decision[updateME][0][currentServer] = 1
            self.decision[updateME][1][previousBank] = 0
            self.decision[updateME][1][currentBank] = 1
        if(case5):
            self.offloaderOfServer[previousServer].remove(updateME)
            self.loanerOfBank[previousBank].remove(updateME)
            self.decision[updateME][0][previousServer] = 0
            self.decision[updateME][1][previousBank] = 0

        # 更新之后获取potential function的值
        potentialAfterUpdate = self.getPotentialFucntion()

        # 每次迭代potential function的值
        self.potentialEachIteration.append(potentialAfterUpdate)

        # 每次迭代每个ME的cost
        for i in range(1, self.numOfME + 1):
            if(sum(self.decision[i][0]) == 0): # 本地计算
                self.meCostEachIteration[i].append(self.costOfLocal(i))
            else:
                s = self.decision[i][0].index(1)
                m = self.decision[i][1].index(1)
                # 先将i从previousServer, preciousBank中去掉
                self.offloaderOfServer[s].remove(i)
                self.loanerOfBank[m].remove(i)
                self.meCostEachIteration[i].append(self.costOfOffloading(i, s, m))
                # 将i加入previousServer, preciousBank中
                self.offloaderOfServer[s].append(i)
                self.loanerOfBank[m].append(i)

        # 测试每一个case
        if(False):
            print('ΔC and ΔΦ：{0}, {1}'.format(update[2], (potentialBeforeUpdate - potentialAfterUpdate) * self.task[updateME][1]))

        self.intertionTime += 1

        # 计算每次迭代的system cost
        self.systemCostEachIteration.append(self.getTotalCostOfAlgorithm())

        return False


    def proposedAlgorithm(self):
        while(self.goToNE() == False):
            pass
        print('Iteration time is : {0}'.format(self.intertionTime))


    def randomDecision(self):
        '''随机决策'''

        for i in range(1, self.numOfME + 1):
            random_num = rd.randint(0, 1)
            if(random_num == 0):
                pass
            else:
                s = rd.randint(1, self.numOfServer)
                m = rd.randint(1, self.numOfBank)
                self.decision[i][0][s] = 1
                self.decision[i][1][m] = 1
                self.offloaderOfServer[s].append(i)
                self.loanerOfBank[m].append(i)
            # random_num = rd.randint(0, self.numOfServer * self.numOfBank)
            # if (random_num == 0):
            #     pass
            # else:
            #     m = math.floor((random_num - 1) / self.numOfServer) + 1
            #     s = random_num - (m - 1) * self.numOfServer
            #     device.decision[i][0][s] = 1
            #     device.decision[i][1][m] = 1
            #     device.offloaderOfServer[s].append(i)
            #     device.loanerOfBank[m].append(i)


    def initPosition(self):
        '''初始化位置'''

        # 初始化server的位置
        for i in range(1, self.numOfServer + 1):
            x = rd.uniform(0, Device.REGION)
            y = rd.uniform(0, Device.REGION)
            self.positionOfServer[i].append(x)
            self.positionOfServer[i].append(y)

        # 初始化ME的位置
        for i in range(1, self.numOfME + 1):
            x = rd.uniform(0, Device.REGION)
            y = rd.uniform(0, Device.REGION)
            self.positionOfME[i].append(x)
            self.positionOfME[i].append(y)

        # 计算ME和server之间的距离
        for i in range(1, self.numOfME + 1):
            xi = self.positionOfME[i][0]
            yi = self.positionOfME[i][1]
            self.distance[i].append(0)
            for j in range(1, self.numOfServer + 1):
                xj = self.positionOfServer[j][0]
                yj = self.positionOfServer[j][1]
                self.distance[i].append(math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)))

        # 计算ME和server之间的传输速率
        N = 100 # dbm
        for i in range(1, self.numOfME + 1):
            self.linkRate[i].append(0)
            for j in range(1, self.numOfServer + 1):
                S = Device.DTP * 0.5 * 14 *  math.pow(self.distance[i][j], -1) * math.pow(35, 2)
                rate = Device.BANDWIDTH * math.log(1 + S/N, 2) / (8 * 1024)
                self.linkRate[i].append(rate)

# device = Device(numOfServer = 5, numOfBank = 3, numOfME = 20, taskSize = [200, 600],
#                 taskCycle = [0.6, 1.5], computationPowerOfServer = [20, 60], computationPowerOfME = [0.8, 1.2],
#                 coEfficient = [4, 0.0025])
# device.initPosition()
# device.proposedAlgorithm()
# device.getInterestRateOfBank()
# print('The interest rate of each bank:{0}'.format(device.interestRateOfBank))
# print('System cost of all local:{}'.format(device.getTotalCostOfLocal()))
# print('System cost of algorithm:{}'.format(device.getTotalCostOfAlgorithm()))
#
# device.restoreState()
# device.randomDecision()
# print('System cost of random:{}'.format(device.getTotalCostOfAlgorithm()))





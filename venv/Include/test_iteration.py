from main import Device
from GA import GA
import pandas as pd
import math


#-------------一张图(迭代次数)--------------


round = 100

numOfMEArr = [20, 30, 40, 50, 60, 70, 80, 90]
len_numOfMEArr = len(numOfMEArr)
numOfAP = [5, 6, 7, 8]
numOfBank = [3, 3, 4, 4]
coefficient = [[[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]],
               [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005], [5, 0.004]],
               [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005], [5, 0.004], [6, 0.003]],
               [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005], [5, 0.004], [6, 0.003], [7, 0.002]]]
len_numOfAP = len(numOfAP)



all_all_numOfIteration = []

for r in range(round):

    all_numOfIteration = []

    for j in range(len_numOfAP):

        numOfIteration = []

        for i in range(len_numOfMEArr):

            device = Device(numOfServer=numOfAP[j], numOfBank=numOfBank[j], numOfME=numOfMEArr[i], taskSize=[600, 1200],
                            taskCycle=[0.8, 1.5], computationPowerOfServer=[8, 16], computationPowerOfME=[0.5, 1],
                            coEfficient=coefficient[j])

            device.initPosition()
            device.proposedAlgorithm1()
            print('Iteration times of algorithm:{}'.format(device.intertionTime))

            numOfIteration.append(device.intertionTime)

        all_numOfIteration.append(numOfIteration)

    all_all_numOfIteration.append(all_numOfIteration)

result = []
for i in range(len_numOfAP):
    temp = []
    for j in range(len_numOfMEArr):
        sum = 0
        for k in range(round):
            sum = sum + all_all_numOfIteration[k][i][j]
        sum = sum / round
        temp.append(sum)
    result.append(temp)



writer = pd.ExcelWriter(r'numberresult/numberOfIteration.xlsx')
df_iteration = pd.DataFrame(data = {'5': result[0], '10': result[1], '15': result[2], '20': result[3]})
df_iteration.to_excel(writer, 'sheet1')
writer.save()
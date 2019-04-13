from main import Device
import pandas as pd


#---------------收敛性-------------------
numOfMEArr = [20, 30, 40, 50, 30]
len_numOfMEArr = len(numOfMEArr)
datacycArr = [1.0, 1.0, 1.0, 1.0, 1.2]
result = []
max_iterationtime = 0

for i in range(len_numOfMEArr):

    device = Device(numOfServer = 8, numOfBank = 3, numOfME = numOfMEArr[i], taskSize = [600, 1200],
                    taskCycle = [datacycArr[i], datacycArr[i]], computationPowerOfServer = [18, 28], computationPowerOfME = [0.5, 1],
                    coEfficient = [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]])

    device.initPosition()
    device.proposedAlgorithm1()
    if(max_iterationtime < device.intertionTime):
        max_iterationtime = device.intertionTime
    result.append(device.beneficialME)

for i in range(len_numOfMEArr):
    size = len(result[i])
    if(max_iterationtime > size):
        for j in range(max_iterationtime - size):
            result[i].append(result[i][size - 1])


writer = pd.ExcelWriter(r'numberresult/timeslot_beneficialME.xlsx')
df_data = pd.DataFrame(data = {'20': result[0], '30': result[1], '40': result[2], '50': result[3], '60': result[4]})
df_data.to_excel(writer, 'sheet1')
writer.save()
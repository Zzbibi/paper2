from main import Device
import pandas as pd


#--------------------迭代过程--------------------


device = Device(numOfServer = 8, numOfBank = 3, numOfME = 25, taskSize = [600, 1200],
                taskCycle = [0.8, 1.5], computationPowerOfServer = [18, 28], computationPowerOfME = [0.5, 1],
                coEfficient = [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]])
device.initPosition()
device.proposedAlgorithm()
meCostEachIteration = device.meCostEachIteration


writer = pd.ExcelWriter(r'numberresult/mecost_eachslot.xlsx')

dictData = {}
for i in range(1, len(meCostEachIteration)):
    key = 'me'+ str(i)
    value = meCostEachIteration[i]
    dictData[key] = value

df_data = pd.DataFrame(data = dictData)
df_data.to_excel(writer, 'sheet1')
writer.save()
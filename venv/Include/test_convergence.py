from main import Device
import pandas as pd


#---------------收敛性-------------------


device = Device(numOfServer = 8, numOfBank = 3, numOfME = 80, taskSize = [600, 1200],
                taskCycle = [0.8, 1.5], computationPowerOfServer = [18, 28], computationPowerOfME = [0.5, 1],
                coEfficient = [[0, 0], [2, 0.0008], [3, 0.001], [4, 0.005]])

device.initPosition()
device.proposedAlgorithm1()
systemCostEachIteration = device.systemCostEachIteration
potentialEachIteration = device.potentialEachIteration

writer = pd.ExcelWriter(r'numberresult/systemcost_potential_eachslot_1.xlsx')
df_data = pd.DataFrame(data = {'systemcost': systemCostEachIteration, 'potential': potentialEachIteration})
df_data.to_excel(writer, 'sheet1')
writer.save()
import numpy
from tqdm import tqdm

train = open("recsys-polimi-2019/challenge2019/dataset/data_train.csv", "r")
train = list(train)[1:]

occurrencies  = numpy.empty(20635, dtype=int)

for i in range(0,20634):
    occurrencies[i] = 0


for line in tqdm(train):
    trackID = int(line.split(",")[1])
    occurrencies[trackID] += 1
    if occurrencies[trackID] == 1785:
        print (trackID)


occurrencies.sort()
print (occurrencies[19800:20634])




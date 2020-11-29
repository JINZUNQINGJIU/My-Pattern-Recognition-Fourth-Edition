import math
import numpy as np
from recognize3 import txt_read
import matplotlib.pyplot as plt
#基于IRIS数据集前40个样本做训练集的剪辑近邻法
filename = 'data.txt'
mardata = txt_read(filename)
train = np.vstack((mardata[0:40, 4:6], mardata[0:40, 8:10]))  # 获取训练集
m = train.shape
a = np.ones((m[0], 1))
getdata = np.hstack((train, a))
getdata[40:80, 2] = 2
Xcur = getdata
for i in range(0, 5):
    Xn = Xcur
    Xold = Xcur
    Xcur = []
    m = Xn.shape
    mm = np.random.randint(1, 4, size=[m[0], 1])
    Xii = np.hstack((Xn, mm))
    for j in range(1, 4):
        xtest = []
        xtrain = []
        for k in range(0, m[0]):
            if Xii[k, 3] == j:
                xtest.append(Xii[k, :])
        r = (j + 1) % 3
        if r == 0:
            r = 3
        for k in range(0, m[0]):
            if Xii[k, 3] == r:
                xtrain.append(Xii[k, :])
        xxtext = np.array(xtest)
        xxtrain = np.array(xtrain)
        xxtext_leng = xxtext.shape
        xxtrain_leng = xxtrain.shape
        if xxtrain_leng[0] == 0:
            break
        finaltext = []
        for g in range(0, xxtext_leng[0]):
            distance = []

            for h in range(0, xxtrain_leng[0]):
                dis = math.sqrt((xxtext[g, 0] - xxtrain[h, 0]) ** 2 + (xxtext[g, 1] - xxtrain[h, 1]) ** 2)
                distance.append(dis)
            mindis = distance.index(min(distance))
            if xxtext[g, 2] == 1 and xxtrain[mindis, 2] == 2:
                xxtext[g, 2] = 0
            elif xxtext[g, 2] == 2 and xxtrain[mindis, 2] == 1:
                xxtext[g, 2] = 0
            if xxtext[g, 2] != 0:
                finaltext.append(xxtext[g, :])
    Xcur = np.array(finaltext)
    sizeXcur = Xcur.shape
    if sizeXcur[0] == m[0]:
        break
test1 = mardata[40:50, 4:6]
test2 = mardata[40:50, 8:10]
test = np.vstack((test1, test2))
y = test.shape
tr = np.ones((y[0], 1))
test = np.hstack((test, tr))
test[10:20, 2] = 2
wrongnum = 0
for i in range(0, y[0]):
    distance1 = []
    for j in range(0, sizeXcur[0]):
        dis1 = math.sqrt((test[i, 0] - Xcur[j, 0]) ** 2 + (test[i, 1] - Xcur[j, 1])**2)
        distance1.append(dis1)
    mindis = distance1.index(min(distance1))
    if test[i, 2] == 1 and Xcur[mindis, 2] == 2:
        print('第{}个数判断错误'.format(i+1))
        wrongnum += 1
    elif test[i,2]==2 and Xcur[mindis,2]==1:
        print('第{}个数判断错误'.format(i + 1))
        wrongnum+=1
    else:
        print('第{}个数判断正确'.format(i+1))
print('错误率是', wrongnum / 20)
Xcur1=[]
Xcur2=[]
for i in range(0,sizeXcur[0]):
    if Xcur[i,2]==1:
        Xcur1.append(Xcur[i,:])
    elif Xcur[i,2]==2:
        Xcur2.append(Xcur[i,:])
Xcur1=np.array(Xcur1)
Xcur2=np.array(Xcur2)
plt.figure(1)
plt.scatter(Xcur1[:,0],Xcur1[:,1],marker='v',label='secondtrain')
plt.scatter(Xcur2[:,0],Xcur2[:,1],marker='x',label='thirdtrain')
plt.scatter(test[0:10,0],test[0:10,1],marker='+',label='secondtest')
plt.scatter(test[10:20,0],test[10:20,1],marker='o',label='secondtest')
plt.xlabel("sepallength/cm")
plt.ylabel("spealwidth/cm")
plt.title('the planar display of IRIS former')
plt.legend()
plt.show()
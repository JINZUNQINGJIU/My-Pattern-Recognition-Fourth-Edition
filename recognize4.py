import math
import numpy as np
import matplotlib.pyplot as plt


def txt_read(filename):
    '''
    读取文件
    :param filename: 文件路径
    :return:
    '''
    import numpy as np
    import re
    pos = []
    file_to_read = open(filename, 'r')
    lines = file_to_read.readlines()  # 整行读取数据
    for line in lines:
        for i in line.split('\n'):
            p_tmp = str(i)
            p_tmp.splitlines()  # 删除换行符
            if p_tmp != '' and p_tmp != '\r':
                pos.append(p_tmp)
    file_to_read.close()
    m = len(pos)
    data = []
    for i in range(0, m):
        M = re.split('[,  ]', pos[i])
        mytest = [j for j in M if j != '']
        n = len(mytest)
        numbers = [float(x) for x in mytest]
        data.append(numbers)
    mardata = np.array(data)
    return mardata


def KNN(list, k):
    minlist = []
    for i in range(1, k + 1):
        mindis = list.index(min(list))
        list[mindis] = max(list)
        minlist.append(mindis)
    return minlist


filename = 'data.txt'
mardata = txt_read(filename)
if __name__ == '__main__':
    Iris_setosa_Sepals_length = mardata[10:50, 0]
    Iris_setosa_Sepals_width = mardata[10:50, 1]
    Iris_versicolor_Sepals_length = mardata[10:50, 4]
    Iris_versicolor_Sepals_width = mardata[10:50, 5]
    Iris_virginica_Sepals_length = mardata[10:50, 8]
    Iris_virginica_Sepals_width = mardata[10:50, 9]
    Iris_setosa_Sepals_tsetl = mardata[0:10, 0]
    Iris_setosa_Sepals_tsetw = mardata[0:10, 1]
    Iris_versicolor_testl = mardata[0:10, 4]
    Iris_versicolor_testw = mardata[0:10, 5]
    Iris_virginica_testl = mardata[0:10, 8]
    Iris_virginica_testw = mardata[0:10, 9]
    train_length = np.hstack((Iris_setosa_Sepals_length, Iris_versicolor_Sepals_length, Iris_virginica_Sepals_length))
    train_width = np.hstack((Iris_setosa_Sepals_width, Iris_versicolor_Sepals_width, Iris_virginica_Sepals_width))
    test_length = np.hstack((Iris_setosa_Sepals_tsetl, Iris_versicolor_testl, Iris_virginica_testl))
    test_width = np.hstack((Iris_setosa_Sepals_tsetw, Iris_versicolor_testw, Iris_virginica_testw))
    dis = []
    wrongnum = 0
    k = 1  # KNN算法类别
    m = len(train_length)
    n = len(test_width)
    for i in range(0, n):
        dis = []
        for j in range(0, m):
            distance = math.sqrt((train_length[j] - test_length[i]) ** 2 + (train_width[j] - test_width[i]) ** 2)
            dis.append(distance)
        # mindis = KNN(dis, k)
        a=min(dis)
        mindis = dis.index(min(dis))
        if mindis <= 39:
            print('第{}个测试样本是第一类花'.format(i + 1))
            if i >= 10:
                wrongnum += 1
                print('判断错误')
        elif mindis > 39 and mindis <= 80:
            print('第{}个测试样本是第二类花'.format(i + 1))
            if i < 10 or i >= 20:
                wrongnum += 1
                print('判断错误')
        else:
            print('第{}个测试样本是第三类花'.format(i + 1))
            if i < 20:
                wrongnum += 1
                print('判断错误')
    print('错误率是', wrongnum / 30)
    plt.figure(1)
    plt.scatter(Iris_setosa_Sepals_length, Iris_setosa_Sepals_width, marker='v', color='black', label='firsttrain')
    plt.scatter(Iris_versicolor_Sepals_length, Iris_versicolor_Sepals_width, marker="x", color="red",
                label='secondtrain')
    plt.scatter(Iris_virginica_Sepals_length, Iris_virginica_Sepals_width, marker="o", color="orange",
                label='thirdtrain')
    plt.scatter(Iris_setosa_Sepals_tsetl, Iris_setosa_Sepals_tsetw, marker='h', color='pink', label='firsttest')
    plt.scatter(Iris_versicolor_testl, Iris_versicolor_testw, marker="+", color="blue", label="secondtest")
    plt.scatter(Iris_virginica_testl, Iris_virginica_testw, marker="d", color="green", label="thirdtest")
    plt.xlabel("sepallength/cm")
    plt.ylabel("spealwidth/cm")
    # plt.xlim((75, 250))
    # plt.ylim((75, 100))
    plt.title('the planar display of IRIS')
    plt.legend()
    plt.show()

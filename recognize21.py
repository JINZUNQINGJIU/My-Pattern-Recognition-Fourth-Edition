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
            p_tmp.splitlines()    #删除换行符
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
def K_near(dist):
    pass

filename = 'data.txt'
mardata = txt_read(filename)
if __name__ == '__main__':
    Iris_versicolor_Sepals_length = mardata[10:50, 4]
    # Iris_versicolor_Sepals_length=Iris_versicolor_Sepals_length[[:,5]
    Iris_versicolor_Sepals_width = mardata[10:50, 5]
    Iris_virginica_Sepals_length = mardata[10:50, 8]
    Iris_virginica_Sepals_width = mardata[10:50, 9]
    Iris_versicolor_testl = mardata[0:10, 4]
    Iris_versicolor_testw = mardata[0:10, 5]
    Iris_virginica_testl = mardata[0:10, 8]
    Iris_virginica_testw = mardata[0:10, 9]
    train_length = np.hstack((Iris_versicolor_Sepals_length, Iris_virginica_Sepals_length))
    # print(train)
    train_width = np.hstack((Iris_versicolor_Sepals_width, Iris_virginica_Sepals_width))
    test_length = np.hstack((Iris_versicolor_testl, Iris_virginica_testl))
    test_width = np.hstack((Iris_versicolor_testw, Iris_virginica_testw))
    # print(test)

    wrongnum = 0
    for i in range(0, 20):
        dis = []
        for j in range(0, 80):
            distance = math.sqrt((train_length[j] - test_length[i]) ** 2 + (train_width[j] - test_width[i]) ** 2)
            dis.append(distance)
        mindis = dis.index(min(dis))
        if mindis <= 39:
            print('这是第二类花')
            if i >= 10:
                wrongnum += 1
        else:
            print('这是第三类花')
            if i < 10:
                wrongnum += 1
    print('错误率是', wrongnum / 20)
    plt.figure(1)
    plt.scatter(Iris_versicolor_Sepals_length, Iris_versicolor_Sepals_width, marker="x", color="red",
                label='secondtrain')
    plt.scatter(Iris_virginica_Sepals_length, Iris_virginica_Sepals_width, marker="o", color="orange",
                label='thirdtrain')
    plt.scatter(Iris_versicolor_testl, Iris_versicolor_testw, marker="+", color="blue", label="secondtest")
    plt.scatter(Iris_virginica_testl, Iris_virginica_testw, marker="d", color="green", label="thirdtest")
    plt.xlabel("sepallength/cm")
    plt.ylabel("spealwidth/cm")
    # plt.xlim((75, 250))
    # plt.ylim((75, 100))
    plt.legend()
    plt.show()

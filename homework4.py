import numpy as np
import matplotlib.pyplot as plt

xtrain = np.array([[220, 90], [240, 95], [220, 95], [180, 95], [140, 90]])
ytrain = np.array([[80, 85], [85, 80], [85, 85], [82, 80], [78, 80]])
test = np.array([[180, 90], [210, 90], [140, 90], [90, 80], [78, 80]])
plt.figure()
plt.scatter(xtrain[:,0],xtrain[:,1],marker='v',color='blue',label='xtrain')
plt.scatter(ytrain[:,0],ytrain[:,1],marker='*',color='pink',label='ytrain')
train1 = np.ones((5, 1))
train2 = train1 * -1
yytrain = ytrain * train2
xxtrain = np.hstack((xtrain, train1))
yytrain = np.hstack((yytrain, train1 * -1))
train = np.vstack((xxtrain, yytrain))
w0 = np.ones((3, 1))
mr = 0
ww = np.zeros((3, 1))
while True:
    tt = w0
    for j in range(0, 10):

        m = 0
        for k in range(0, 3):
            a = w0[k, 0] * train[j, k]
            m += a
        mr += 1
        if m <= 0:
            for h in range(0, 3):
                ww[h, 0] = train[j, h]
            w0 = w0 + ww
        x = np.linspace(-20, 10)
        y = -w0[0, 0] / w0[1, 0] * x - w0[2, 0] / w0[1, 0]
        plt.plot(x, y, color='green')
    if tt[0, 0] == w0[0, 0] and tt[1, 0] == w0[1, 0] and tt[2, 0] == w0[2, 0]:
        break
x = np.linspace(-10, 250)
y = -w0[0, 0] / w0[1, 0] * x - w0[2, 0] / w0[1, 0]
plt.xlim((-20, 250))
plt.ylim((-15, 120))
plt.plot(x, y, color='red',label='final line')
plt.xlabel('xlabelf')
plt.ylabel('ylabelf')
plt.legend()
plt.title('recognize')
plt.show()

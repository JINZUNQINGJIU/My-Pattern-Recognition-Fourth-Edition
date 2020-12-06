import numpy as np
import matplotlib.pyplot as plt
import sys

def Sigmod(input):
    y = 1 / (1 + np.exp(-input))
    return y


def disiggmod(y):
    return y * (1 - y)


class Bp_Neural_network(object):
    def __init__(self, input, hidden, output, learnrate=0.1):
        self.input = input
        self.output = output
        self.hidden = hidden
        self.learnrate = learnrate
        self.w0 = np.random.randn(self.input, self.hidden)  # 输入隐藏层权重初始值
        self.w1 = np.random.randn(self.hidden, self.output)  # 隐藏输出层权重初始值
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
        self.ai = np.zeros((self.input, 1))  # 输入阈值初始
        self.ah = np.ones((self.hidden, 1))  # 隐藏层阈值初始值
        self.ao = np.ones((self.output, 1))  # 输出层阈值初始值
        self.hidout = np.zeros((self.hidden, 1))  # 隐藏层输出
        self.outo = np.zeros((self.output, 1))  # 输出层输出
        self.outoo = np.zeros((self.output, 1))  # 输出线性化前

    def feedword(self, inputs):
        # 前向传播
        # if len(inputs) != self.input:
        #     raise ValueError('Wrong number of inputs')
        for i in range(0, self.input):
            self.ai[i, 0] = inputs[i]
        for j in range(0, self.hidden):
            sum = 0.0
            for i in range(0, self.input):
                sum += self.ai[i, 0] * self.w0[i, j] + self.ah[j, 0]
            self.hidout[j, 0] = Sigmod(sum)
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.hidout[j, 0] * self.w1[j, k] + self.ao[k, 0]
            self.outoo[k, 0] = sum
            self.outo[k, 0] = Sigmod(sum)
        return self.outo

    def feedword1(self, inputs):
        # 测试用前向传播
        # if len(inputs) != self.input:
        #     raise ValueError('Wrong number of inputs')
        for i in range(0, self.input):
            self.ai[i, 0] = inputs[i]
        for j in range(0, self.hidden):
            sum = 0.0
            for i in range(0, self.input):
                sum += self.ai[i, 0] * self.w0[i, j] + self.ah[j, 0]
            self.hidout[j, 0] = Sigmod(sum)
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.hidout[j, 0] * self.w1[j, k] + self.ao[k, 0]
            self.outoo[k, 0] = sum
            self.outo[k, 0] = Sigmod(sum)
        return self.outoo

    def backforward(self, realnum):
        output_deltas = np.zeros((self.output, 1))
        for k in range(0, self.output):
            error = realnum[k] - self.outo[k, 0]
            output_deltas[k, 0] = disiggmod(self.outo[k, 0]) * error  # 计算输出层阈值变化gi
        hidden_deltas = np.zeros((self.hidden, 1))
        for j in range(self.hidden):
            error = 0
            for k in range(0, self.output):
                error += output_deltas[k, 0] * self.w1[j, k]  # 计算隐层和输出层之间的权值变化
            hidden_deltas[j, 0] = disiggmod(self.hidout[j, 0]) * error  # eh
        for j in range(0, self.hidden):
            for k in range(0, self.output):
                self.w1[j, k] += self.learnrate * output_deltas[k, 0] * self.outo[k, 0]  # 隐藏输出层权重更新
        for k in range(0, self.output):
            self.ao[k, 0] -= self.learnrate * output_deltas[k, 0]  # 输出层阈值更新
        for i in range(0, self.input):
            for j in range(0, self.hidden):
                self.w0[i, j] += self.learnrate * hidden_deltas[j, 0] * self.ai[i, 0]  # 输入隐藏层权重更新
        for j in range(0, self.hidden):
            self.ah[j, 0] -= self.learnrate * hidden_deltas[j, 0]  # 隐藏层阈值更新
        errorr = 0.0
        for k in range(len(realnum)):
            errorr += 0.5 * (realnum[k] - self.ao) ** 2
        return errorr

    def train(self, patterns, level, iterations=10000):
        for i in range(0, iterations):

            for j in range(0, level):
                error = 0.0
                inputs = patterns[j, 0]
                inputs = np.array([inputs])
                targets = patterns[j, 1]
                targets = np.array([targets])
                self.feedword(inputs)
                errorw = self.backforward(targets)
                error += errorw[0,0]
                if error < 0.0001:
                    sys.exit(0)
            if i % 2000 == 0:
                print('error{}'.format(error))

    def predict(self, X):
        prediction = self.feedword1(X)
        return prediction


if __name__ == '__main__':
    xtrain = np.linspace(0, 2, 21)
    #xtrain = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    ytrain = (xtrain ** 2)/4
    #ytrain = np.array([1, 1, 0, 0])
    train = np.vstack((xtrain, ytrain))
    train = np.transpose(train)

    NN = Bp_Neural_network(1, 3, 1, learnrate=0.7)
    NN.train(train, 21)
    # test=[1.23]
    # mm = NN.predict(test)

#手写单个神经元实现 and 算法


class Nerve:
    def __init__(self, leraning_rate):
        #初始化两个权重为0
        self.w1 = 0.0
        self.w2 = 0.0
        #偏移量初始化为0
        self.bias = 0.0
        #初始化学习率
        self.leraning_rate = leraning_rate

    #设置激活函数
    def actevite(self, y):
        return 1 if \
            y > 0 else 0

    #训练结果
    def predict(self, x1, x2):
        return self.actevite(x1 * self.w1 + x2 * self.w2 + self.bias)

    #梯度下降法更新参数
    def updata(self, delta, x1, x2):
        self.w1 = self.w1 + delta * self.leraning_rate * x1
        self.w2 = self.w2 + delta * self.leraning_rate * x2
        self.bias += delta * self.leraning_rate
        print('w1 = ', self.w1, 'w2 = ', self.w2, 'bias = ', self.bias)

    #训练入口
    def train(self, X_train, labels):
        data = zip(X_train, labels)
        for (x, label) in data:
            x1, x2 = x
            self._one_train(x1, x2, label)

    #训练一组参数
    def _one_train(self, x1, x2, label):
        y = self.predict(x1, x2)
        delta = label - y
        self.updata(delta, x1, x2)


def SetData(X_train, labels, leraning_rate, epoch):
    N = Nerve(leraning_rate)
    for n in range(epoch):
        N.train(X_train, labels)
    return N


X_train = [[1, 0], [0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]
labels = [0, 0, 1, 0, 0, 0]
leraning_rate = 0.1
epoch = 10
#开始训练
p = SetData(X_train, labels, leraning_rate, epoch)
#此时P为训练好的模型
y1 = p.predict(0, 1)
y2 = p.predict(1, 1)
y3 = p.predict(0, 0)
print('0 and 1 = ', y1)
print('1 and 1 = ', y2)
print('0 and 0 = ', y3)

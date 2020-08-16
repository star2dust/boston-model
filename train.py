import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # 读入训练数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')
    # print(data)
    
    # 读入之后的数据被转化成1维array，其中array的
    # 第0-13项是第一条数据，第14-27项是第二条数据，.... 
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    
    # 这里对原始数据做reshape，变成N x 14的形式
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    
    # 取80%的数据作为训练集，预留20%的数据用于测试模型的预测效果
    #（训练好的模型预测值与实际房价的差距）。打印训练集的形状可见，
    # 我们共有404个样本，每个样本含有13个特征和1个预测值。
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # print(training_data.shape)
    
    # 对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。
    # 这样做有两个好处：
    # 1. 模型训练更高效。
    # 2. 特征前的权重大小可代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。
    
    # 计算train数据集的最大值，最小值，平均值
    # \是python的分行符，相当于matlab的...
    # 注意：\后面一个空格都不能有
    maximums, minimums, avgs = \
                         training_data.max(axis=0), \
                         training_data.min(axis=0), \
         training_data.sum(axis=0) / training_data.shape[0]
    # 对所有数据进行归一化处理(-1,1)
    # 测试样本也用训练样本的最大最小值进行归一化
    # 因为测试模拟的是真实环境，但是真实数据无法获得只能用训练数据
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    
    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 构建线性回归神经网络（没有激活函数）
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1) 
        self.b = 0.
        
    def forward(self, x):
        # dot里面为长度相同的行向量
        # x: mxn, w: nx1
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        # MSE
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        # 维度(n,)=>(n,1)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        
    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        # 每次从总的数据集中随机抽取出小部分数据来代表整体，基于这部分数据计算梯度和损失，然后更新参数。
        # 这种方法被称作小批量随机梯度下降法（Mini-batch Stochastic Gradient Descent），简称SGD
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        return losses
    
# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epoches=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

# 写在前面

本文源于百度AI平台飞桨学院《[百度架构师手把手带你零基础实践深度学习](https://aistudio.baidu.com/aistudio/course/introduce/1297)》课程中我自己的心得和理解。

本文旨在介绍使用飞桨框架构建神经网络过程，并从房价预测模型的理解和代码的构建角度来整理所学内容，不求详尽但求简洁明了。

# 模型构建基本流程

飞桨的模型覆盖计算机视觉、自然语言处理和推荐系统等主流应用场景，所有场景的代码结构完全一致，如[图1](#nnflow)所示。

<a name="nnflow"></a> 
<img src="figures\1597554589992.png" alt="nnflow" width="450" />

<center>

图1. 使用飞桨框架构建神经网络过程
</center>

# 飞桨重写房价预测模型

数据处理之前，需要先加载飞桨框架的相关类库。
```python
#加载飞桨、Numpy和相关类库
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
```

## 1. 数据处理
数据处理包含五个部分：数据导入、数据形状变换、数据集划分、数据归一化处理和封装load data函数。数据预处理后，才能被模型调用。数据处理的代码不依赖paddle框架实现，使用numpy库即可。

对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。这样做有两个好处：
- 模型训练更高效。
- 特征前的权重大小可代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。

```python
def load_data():
    # 从文件读入训练数据
    datafile = './work/housing.data'
    # 数据以空格分隔
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    # \是python的分行符，相当于matlab的...
    # 注意：\后面一个空格都不能有
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对全部数据进行归一化处理，包括train和test
    # 测试样本也用训练样本的最大最小值进行归一化
    # 因为测试模拟的是真实环境，但是真实数据无法获得只能用训练数据
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 数据归一化后按比例重新划分训练集和测试集
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data
```

## 2. 模型设计
模型定义的实质是定义线性回归的网络结构，创建`Regressor`类，定义`init`函数和`forward`函数。
```python
class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor, self).__init__() 
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=13, output_dim=1, act=None)
    
    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x
```

## 3. 训练配置
训练配置过程包含四步，如[图2](#trainconfig)所示：

<a name="trainconfig"></a> 

<img src="figures\1597555885064.png" alt="trainconfig" width="450" />

<center>
图2. 训练配置流程示意图
</center>

- 以guard函数指定运行训练的机器资源，表明在with作用域下的程序均执行在本机的CPU资源上。dygraph.guard表示在with作用域下的程序会以飞桨动态图的模式执行（实时执行）。
- 声明定义好的回归模型Regressor实例，并将模型的状态设置为训练。
- 使用load_data函数加载训练数据和测试数据。
- 设置优化算法和学习率，优化算法采用随机梯度下降SGD，学习率设置为0.01。

训练配置代码如下所示：
```python
# 定义飞桨动态图的工作环境
with fluid.dygraph.guard(fluid.CPUPlace()):
    # 声明定义好的线性回归模型
    model = Regressor()
    # 开启模型训练模式
    model.train()
    # 加载数据
    training_data, test_data = load_data()
    # 定义优化算法，这里使用随机梯度下降-SGD
    # 学习率设置为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())
```
## 4. 训练过程

对于一个数据集，我们遍历它一次称为一个epoch。对于一次遍历，我们分批次读取数据，一批称为一个batch。训练过程采用二层循环嵌套方式：
- 内层循环： 负责整个数据集的一次遍历，采用分批次方式。
- 外层循环： 定义遍历数据集的次数，通过参数EPOCH_NUM设置。

假设数据集样本数量为1000，一个批次有10个样本，则遍历一次数据集的批次数量是1000/10=100，即内层循环需要执行100次。若数据集需要使用5次，则总循环数为5*100=500。

每次内层循环都需要执行如下四个步骤，如 [图3](#innerloop) 所示
<a name="innerloop"></a> 

<img src="figures\1597556938631.png" alt="innerloop" width="450" />

<center>

图3. 内循环计算过程
</center>

- 数据准备：将一个批次的数据转变成np.array和内置格式。
- 前向计算：将一个批次的样本数据灌入网络中，计算输出结果。
- 计算损失函数：以前向计算结果和真实房价作为输入，通过损失函数square_error_cost计算出损失函数值（Loss）。
- 反向传播：执行梯度反向传播backward函数，即从后到前逐层计算每一层的梯度，并根据设置的优化算法更新参数opt.minimize。

```python
# 定义飞桨动态图工作环境
with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 10   # 设置外层循环次数
    BATCH_SIZE = 10  # 设置batch大小
    
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype('float32') # 获得当前批次训练数据
            y = np.array(mini_batch[:, -1:]).astype('float32') # 获得当前批次训练标签（真实房价）
            # 将numpy数据转为飞桨动态图variable形式
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)
            
            # 前向计算
            predicts = model(house_features)
            
            # 计算损失
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()
```

## 5. 保存模型
将模型当前的参数数据model.state_dict()保存到文件中（通过参数指定保存的文件名 LR_model），以备预测或校验的程序调用，代码如下所示。

```python
# 定义飞桨动态图工作环境
with fluid.dygraph.guard(fluid.CPUPlace()):
    # 保存模型参数，文件名为LR_model
    fluid.save_dygraph(model.state_dict(), 'LR_model')
    print("模型保存成功，模型参数保存在LR_model中")
```
为什么是先保存模型，再加载模型呢？这是因为在实际应用中，训练模型和使用模型往往是不同的场景。模型训练通常使用大量的线下服务器（不对外向企业的客户/用户提供在线服务），而模型预测则通常使用线上提供预测服务的服务器，或者将已经完成的预测模型嵌入手机或其他终端设备中使用。因此先保存再加载方式更贴合真实场景的使用方法。

# 测试模型预测效果

下面我们选择一条数据样本，测试下模型的预测效果。测试过程和在应用场景中使用模型的过程一致，主要可分成如下三个步骤：

- 配置模型预测的机器资源。本案例默认使用本机，因此无需写代码指定。
- 将训练好的模型参数加载到模型实例中。由两个语句完成，第一句是从文件中读取模型参数；第二句是将参数内容加载到模型。加载完毕后，需要将模型的状态调整为eval()（校验）。上文中提到，训练状态的模型需要同时支持前向计算和反向传导梯度，模型的实现较为臃肿，而校验和预测状态的模型只需要支持前向计算，模型的实现更加简单，性能更好。
- 将待预测的样本特征输入到模型中，打印输出的预测结果。

从test_data中抽一条样本作为测试样本，具体实现代码如下所示。
```python
# 定义飞桨动态图的工作环境
with fluid.dygraph.guard(fluid.CPUPlace()):
    # 测试模型 参数为保存模型参数的文件地址
    model_dict, _ = fluid.load_dygraph('LR_model')
    model.load_dict(model_dict)
    model.eval()
    # 随机抽取一条测试数据
    np.random.shuffle(test_data)
    test_batch = test_data[0:1]
    # 将数据转为动态图的variable格式
    x = np.array(test_batch[:, :-1]).astype('float32') # 获得当前批次训练数据
    y = np.array(test_batch[:, -1:]).astype('float32') # 获得当前批次训练标签（真实房价）
    # 将numpy数据转为飞桨动态图variable形式
    house_features = dygraph.to_variable(x)
    x = dygraph.to_variable(house_features)
    results = model(house_features)

    # 对结果做反归一化处理
    results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
    label = y * (max_values[-1] - min_values[-1]) + avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(results.numpy(), label))
```
通过比较“模型预测值”和“真实房价”可见，模型的预测效果与真实房价接近。

# 源代码

本文所有内容的源代码已上传至[我的GitHub](https://github.com/star2dust/boston-model)。其中`train.py`是房价预测模型的纯numpy实现，`pdtrain.py`是paddlepaddle实现，大家可以对比一下异同。

如果尚未安装numpy和paddlepaddle，请命令行输入如下代码安装：
```shell
pip install -r requirements.txt
```

如果喜欢，欢迎点赞和fork。
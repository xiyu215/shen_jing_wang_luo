import numpy as np
import matplotlib.pyplot as plt
from common.gridworld import GridWorld
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F


# 【补全1】把二维坐标 (y, x) 变成长度为 12 的独热向量，形状为 (1, 12)
def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


# 【补全2】定义 Q 网络模型（大脑）
class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)  # 隐藏层：100 个神经元
        self.l2 = L.Linear(4)    # 输出层：对应上下左右 4 个动作的得分

    def forward(self, x):
        x = F.relu(self.l1(x))   # 用 ReLU 把直线掰弯
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01           # 【修正1】tr 改为 lr
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state)
            return qs.data.argmax()

    # 【修正2】保留完整的 update，删掉下面重复的半成品
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)  # 【修正3】axis=0 改为 axis=1
            next_q.unchain()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()
        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000  # 回合数量
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    # 【修正4】以下全部缩进到 for 循环内部，才能真正训练 1000 局
    while not done:
        action = agent.get_action(state)  # 【修正5】getaction 改为 get_action
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)
    if episode % 100 == 0:
        print(f'episode: {episode}, loss: {float(average_loss):.4f}')

# 1. 画出每个回合的平均损失变化曲线
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# 2. 把神经网络预测的 Q 值整理成字典，调用 common 里的 render_q 画出 Q 值分布与最佳路线
Q = {}
for state in env.states():
    qs = agent.qnet(one_hot(state))
    for action in env.actions():
        Q[state, action] = float(qs.data[0, action])
env.render_q(Q)
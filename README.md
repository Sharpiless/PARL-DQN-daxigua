# 用DQN强化学习算法玩“合成大西瓜”！

视频链接：

https://www.bilibili.com/video/BV1Tz4y1U7HE

https://www.bilibili.com/video/BV1Wy4y1n73E

https://www.bilibili.com/video/BV1gN411d7dr

## 1. 安装依赖库

> 其中游戏代码使用pygame重构

> 物理模块使用pymunk

注：paddlepaddle版本为1.8.0，parl版本为1.3.1


```
# !pip install pygame -i https://mirror.baidu.com/pypi/simple
# !pip install parl==1.3.1 -i https://mirror.baidu.com/pypi/simple
# !pip install pymunk
```


```
# !unzip work/code.zip -d ./
```

## 2. 设置环境变量

> 由于notebook无法显示pygame界面，所以我们设置如下环境变量


```
import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
```

## 3. 构建多层神经网络

> 该版本使用两层全连接层

> 卷积神经网络版本为：https://aistudio.baidu.com/aistudio/projectdetail/1540300


```
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
from parl.utils import logger
import random
import collections

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等


class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 256
        hid2_size = 256
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
```


## 4. 构建DQN算法、Agent和经验池


```
# from parl.algorithms import DQN # 也可以直接从parl库中导入DQN算法
import pygame

class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost



class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)


# 训练一个episode
def run_episode(env, agent, rpm, episode):
    total_reward = 0
    env.reset()
    action = np.random.randint(0, env.action_num - 1)
    obs, _, _ = env.next(action)
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done = env.next(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
        if not step % 20:
            logger.info('step:{} e_greed:{} action:{} reward:{}'.format(
                step, agent.e_greed, action, reward))
        if not step % 500:
            image = pygame.surfarray.array3d(
                 pygame.display.get_surface()).copy()
            image = np.flip(image[:, :, [2, 1, 0]], 0)
            image = np.rot90(image, 3)
            img_pt = os.path.join('outputs', 'snapshoot_{}_{}.jpg'.format(episode, step))
            cv2.imwrite(img_pt, image)
    return total_reward
```

    pygame 2.0.1 (SDL 2.0.14, Python 3.7.4)
    Hello from the pygame community. https://www.pygame.org/contribute.html


## 5. 创建Agent实例


```
from State2NN import AI_Board

env = AI_Board()  
action_dim = env.action_num  
obs_shape = 16 * 13  
e_greed = 0.2

rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape,
    act_dim=action_dim,
    e_greed=e_greed,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低
```

    [32m[02-20 22:45:25 MainThread @machine_info.py:86][0m nvidia-smi -L found gpu count: 1
    [32m[02-20 22:45:25 MainThread @machine_info.py:86][0m nvidia-smi -L found gpu count: 1


## 6. 训练模型


```
from State2NN import AI_Board
import os

dirs = ['weights', 'outputs']
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)

# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm, episode=0)

max_episode = 2000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm, episode+1)
        episode += 1
        save_path = './weights/dqn_model_episode_{}.ckpt'.format(episode)
        agent.save(save_path)
        print('-[INFO] episode:{}, model saved at {}'.format(episode, save_path))
        env.reset()

# 训练结束，保存模型
save_path = './final.ckpt'
agent.save(save_path)
```

## 7. 游戏环境补充说明



### 7.1 游戏共有11种水果：

![](https://ai-studio-static-online.cdn.bcebos.com/6758c92fb70e4b59904cd0d39b7e8b4e9c70943c4f9046129f289c8e8b52ecf5)

### 7.2 碰撞检测：


```python
def setup_collision_handler(self):
        def post_solve_bird_line(arbiter, space, data):
            if not self.lock:
                self.lock = True
                b1, b2 = None, None
                i = arbiter.shapes[0].collision_type + 1
                x1, y1 = arbiter.shapes[0].body.position
                x2, y2 = arbiter.shapes[1].body.position
```

### 7.3 奖励机制：

每合成一种水果，reward加相应的分数


| 水果 | 分数 |
| -------- | -------- |
| 樱桃     | 2     |
| 橘子     | 3     |
| ...     | ...     |
| 西瓜     | 10     |
| 大西瓜     | 100     |

```python
if i < 11:
	self.last_score = self.score
	self.score += i
elif i == 11:
	self.last_score = self.score
	self.score += 100
```


### 7.4 惩罚机制:

如果一次action后 1s（即新旧水果生成间隔）没有成功合成水果，则reward减去放下水果的分数

```python
_, reward, _ = self.next_frame(action=action)
for _ in range(int(self.create_time * self.FPS)):
	_, nreward, _ = self.next_frame(action=None, generate=False)
	reward += nreward
	if reward == 0:
		reward = -i
```

### 7.5 输入特征：

之前的版本(https://aistudio.baidu.com/aistudio/projectdetail/1540300)输入特征为游戏截图，采用ResNet提取特征

但是直接原图输入使得模型很难学习到有效的特征

因此新版本使用pygame接口获取当前状态

```python
def get_feature(self, N_class=12, Keep=15):
        # 特征工程
        c_t = self.i
        # 自身类别
        feature_t = np.zeros((1, N_class + 1), dtype=np.float)
        feature_t[0, c_t] = 1.
        feature_t[0, 0] = 0.5
        feature_p = np.zeros((Keep, N_class + 1), dtype=np.float)
        Xcs = []
        Ycs = []
        Ts = []
        for i, ball in enumerate(self.balls):
            if ball:
                x = int(ball.body.position[0])
                y = int(ball.body.position[1])
                t = self.fruits[i].type
                Xcs.append(x/self.WIDTH)
                Ycs.append(y/self.HEIGHT)
                Ts.append(t)
        sorted_id = sorted_index(Ycs)
        for i, id_ in enumerate(sorted_id):
            if i == Keep:
                break
            feature_p[i, Ts[id_]] = 1.
            feature_p[i, 0] = Xcs[id_]
            feature_p[i, -1] = Ycs[id_]
            
        image = np.concatenate((feature_t, feature_p), axis=0)
        return image
```

注：N_class = 水果类别数 + 1

#### feature_t：
> 用于表示当前手中水果类别的ont-hot向量；

#### feature_p：

> 用于表示当前游戏状态，大小为(Keep, N_class + 1)

> Keep 表示只关注当前位置最高的 Keep 个水果

> N_class - 1 是某个水果类别的ont-hot向量， 0 位置为 x 坐标，-1 位置为 y 坐标（归一化）

目前用的就是这些个特征。。。

# 张老板来了hahh，老板大气（滑稽.jpg）

## 8. 我的公众号

![](https://ai-studio-static-online.cdn.bcebos.com/581bcc5e56594107859b2a8ccebba0e9938d10759d8242e689ae64680ab94150)


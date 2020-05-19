""" 本代码仅作为DQN模型的参考实现
"""

import torch
import torch.nn as nn
import gym

class DQN(nn.Module):
    def __init__(self, naction, nstate, nhidden):
        super(DQN, self).__init__()
        self.naction = naction
        self.nstate = nstate
        self.linear1 = nn.Linear(naction + nstate, nhidden)
        self.linear2 = nn.Linear(nhidden, nhidden)
        self.linear3 = nn.Linear(nhidden, 1)
    
    def forward(self, state, action):
        action_enc = torch.zeros(action.size(0), self.naction)
        action_enc.scatter_(1, action.unsqueeze(-1), 1)
        output = torch.cat((state, action_enc), dim=-1)
        output = torch.relu(self.linear1(output))
        output = torch.relu(self.linear2(output))
        output = self.linear3(output)
        return output.squeeze(-1)

class Memory(object):
    def __init__(self, capacity=1000):

        self.capacity = capacity
        self.size = 0
        self.data = []
        
    def __len__(self):
        return self.size
        
    def push(self, state, action, state_next, reward, is_ended):
        
        if len(self) > self.capacity:
            k = random.randint(self.capacity)
            self.data.pop(k)
            self.size -= 1
        
        self.data.append((state, action, state_next, reward, is_ended))
        
    def sample(self, bs):
        data = random.choices(self.data, k=bs)
        states, actions, states_next, rewards, is_ended = zip(*data)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        states_next = torch.tensor(states_next, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        is_ended = torch.tensor(is_ended, dtype=torch.float32)
        
        return states, actions, states_next, rewards, is_ended

# 定义两个网络，用于加速模型收敛
dqn = DQN(2, 4, 8)
dqn_t = DQN(2, 4, 8)
dqn_t.load_state_dict(copy.deepcopy(dqn.state_dict()))
eps = 0.1
# 折扣系数
gamma = 0.999

optim = torch.optim.Adam(dqn.parameters(), lr=1e-3)
criterion = HuberLoss()         
                      
step_cnt = 0
mem = Memory()

for episode in range(300):
    state = env.reset()
    while True:
        action_t = torch.tensor([0, 1])
        state_t = torch.tensor([state, state], dtype=torch.float32)
        
        # 计算最优策略
        torch.set_grad_enabled(False)
        q_t = dqn(state_t, action_t)
        max_t = q_t.argmax()
        torch.set_grad_enabled(True)
        
        # 探索和利用的平衡
        if random.random() < eps:
            max_t = random.choice([0, 1])
        else:
            max_t = max_t.item()
        
        state_next, reward, done, info = env.step(max_t)
        
        mem.push(state, max_t, state_next, reward, done)
        state = state_next
        
        if done:
            break
    
        # 重放训练
        for _ in range(10):
            state_t, action_t, state_next_t, reward_t, is_ended_t = \
                mem.sample(32)

            q1 = dqn(state_t, action_t)
            
            torch.set_grad_enabled(False)
            q2_0 = dqn_t(state_next_t, 
                         torch.zeros(state_t.size(0), dtype=torch.long))
            q2_1 = dqn_t(state_next_t, 
                         torch.ones(state_t.size(0), dtype=torch.long))
            # 利用Bellman方程进行迭代
            q2_max = reward_t + gamma*(1-is_ended_t)*
                (torch.stack((q2_0, q2_1), dim=1).max(1)[0])
            torch.set_grad_enabled(True)
            # 优化损失函数
            delta = q2_max - q1
            loss = criterion(delta)
            optim.zero_grad()
            loss.backward()
            for p in dqn.parameters(): p.grad.data.clamp_(-1, 1)
            optim.step()          
            step_cnt += 1
                            
            # 同步两个网络的参数
            if step_cnt % 1000 == 0:
                dqn_t.load_state_dict(copy.deepcopy(dqn.state_dict()))
env.close()

import os
import random
import dill as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from tqdm import tqdm

from environment import GridWorld
from subrl.utils.network import append_state
from visualization import Visu
from subpo import calculate_submodular_reward, compute_subpo_advantages

# SAC网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_probs = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_probs(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1_q1 = nn.Linear(state_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, action_dim)
        
        # Q2
        self.fc1_q2 = nn.Linear(state_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # Q1 network
        x1 = F.relu(self.fc1_q1(state))
        x1 = F.relu(self.fc2_q1(x1))
        q1 = self.fc3_q1(x1)
        
        # Q2 network
        x2 = F.relu(self.fc1_q2(state))
        x2 = F.relu(self.fc2_q2(x2))
        q2 = self.fc3_q2(x2)
        
        return q1, q2

class SAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        """更新SAC网络"""
        if len(replay_buffer) < batch_size:
            return
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_dist = Categorical(next_action_probs)
            next_log_probs = next_dist.logits
            
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            # Add entropy term
            next_v = torch.sum(next_action_probs * (next_q - self.alpha * next_log_probs), dim=1)
            target_q = rewards + (1 - dones.float()) * self.gamma * next_v
        
        current_q1, current_q2 = self.critic(states)
        current_q1 = current_q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = current_q2.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.logits
        
        q1, q2 = self.critic(states)
        q = torch.min(q1, q2)
        
        # Policy loss with entropy regularization
        policy_loss = torch.sum(action_probs * (self.alpha * log_probs - q), dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, main_network, target_network):
        """软更新目标网络"""
        for main_param, target_param in zip(main_network.parameters(), target_network.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def select_cell_from_archive(archive, current_step=0, total_steps=1):
    """
    从存档中选择细胞进行探索
    - 前4/5步骤：使用基于步数的概率权重选择（权重 = 1/(时间步 + 1)）
    - 后1/5步骤：使用均匀采样
    """
    if not archive:
        return None, None

    cell_keys = list(archive.keys())
    
    # 检查是否已达到总步数的4/5
    progress_ratio = current_step / total_steps if total_steps > 0 else 0
    use_uniform_sampling = progress_ratio >= 0.8  
    
    if use_uniform_sampling:
        # 使用均匀采样
        normalized_weights = [1.0 / len(cell_keys) for _ in cell_keys]
    else:
        # 使用步数加权采样
        weights = []
        for cell_key in cell_keys:
            time_step = cell_key[0]  # cell key的第一个元素是时间步
            weight = 1.0 / (time_step + 1)  # 步数越少，权重越大
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(cell_keys) for _ in cell_keys]
    
    # 根据权重随机选择cell
    selected_index = np.random.choice(len(cell_keys), p=normalized_weights)
    selected_cell_key = cell_keys[selected_index]
    
    return selected_cell_key, archive[selected_cell_key]

def sample_excellent_trajectories(filepath: str, 
                                  method='top_n', 
                                  n=10, 
                                  p=0.1, 
                                  threshold=0):
    """
    从Go-Explore存档中加载数据并根据指定方法采样高质量轨迹
    """
    # 检查文件是否存在并加载数据
    if not os.path.exists(filepath):
        print(f"错误：存档文件未找到 '{filepath}'")
        return []
    
    try:
        with open(filepath, "rb") as f:
            archive = pickle.load(f)
        if not archive:
            print("警告：存档库为空。")
            return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

    # 提取所有轨迹数据并按奖励排序
    all_trajectories_data = list(archive.values())
    all_trajectories_data.sort(key=lambda x: x['reward'], reverse=True)

    # 根据指定方法进行采样
    sampled_trajectories = []
    if method == 'top_n':
        num_to_sample = min(n, len(all_trajectories_data))
        sampled_trajectories = all_trajectories_data[:num_to_sample]
        print(f"方法: Top-N。从 {len(all_trajectories_data)} 条轨迹中筛选出最好的 {len(sampled_trajectories)} 条。")

    elif method == 'top_p':
        if not (0 < p <= 1):
            print("错误：百分比 'p' 必须在 (0, 1] 之间。")
            return []
        num_to_sample = int(len(all_trajectories_data) * p)
        sampled_trajectories = all_trajectories_data[:num_to_sample]
        print(f"方法: Top-P。从 {len(all_trajectories_data)} 条轨迹中筛选出最好的前 {p*100:.1f}% ({len(sampled_trajectories)} 条)。")

    elif method == 'threshold':
        sampled_trajectories = [data for data in all_trajectories_data if data['reward'] >= threshold]
        print(f"方法: Threshold。从 {len(all_trajectories_data)} 条轨迹中筛选出 {len(sampled_trajectories)} 条奖励不低于 {threshold} 的轨迹。")
        
    else:
        print(f"错误：未知的采样方法 '{method}'。请使用 'top_n', 'top_p', 或 'threshold'。")

    return sampled_trajectories

# --- 核心参数设置 ---
EXPLORATION_STEPS = 100000  # 总探索步数

def run_sac_with_go_explore(env=GridWorld, params=None):
    """
    执行SAC结合Go-Explore的训练
    """
    print("--- SAC with Go-Explore Training 开始 ---")
    
    # 1. 初始化存档库 (Archive)
    archive = {}
    
    # 2. 初始化SAC智能体
    H = params["env"]["horizon"]
    MAX_Ret = 2*(H+1)
    if params["env"]["disc_size"] == "large":
        MAX_Ret = 3*(H+2)
    
    # SAC使用状态历史作为输入，维度为 H-1
    state_dim = H - 1  # 修正状态维度
    action_dim = env.action_dim
    
    sac_agent = SAC(state_dim, action_dim, lr=params["alg"]["lr"])
    replay_buffer = ReplayBuffer(capacity=100000)
    
    env.common_params["batch_size"] = 1
    env.initialize(params["env"]["initial"])
    initial_state_tensor = env.state.clone()
    initial_state_id = initial_state_tensor.item()
    
    # 初始细胞
    initial_cell_key = (0, initial_state_id)
    initial_reward = calculate_submodular_reward([initial_state_tensor], env)
    
    archive[initial_cell_key] = {
        'reward': initial_reward,
        'states': [initial_state_tensor],
        'actions': [],
        'times_selected': 0
    }
    print(f"初始时空细胞加入存档库: Cell {initial_cell_key}, Reward: {initial_reward}")
    
    # 3. 执行N次探索循环
    pbar = tqdm(total=EXPLORATION_STEPS, desc="SAC Go-Exploring")
    for step in range(EXPLORATION_STEPS):
        # 3.1 选择细胞
        cell_key_to_explore_from, selected_cell_data = select_cell_from_archive(
            archive, current_step=step, total_steps=EXPLORATION_STEPS)
        
        if selected_cell_data is None:
            print("错误：存档库为空，无法继续探索。")
            break
            
        archive[cell_key_to_explore_from]['times_selected'] += 1

        # 3.2 前往该细胞状态
        env.initialize(params["env"]["initial"])
        for action in selected_cell_data['actions']:
            env.step(0, torch.tensor([action]))

        # 3.3 使用SAC策略执行剩余步数
        current_states = selected_cell_data['states'][:]
        current_actions = selected_cell_data['actions'][:]
        
        # 计算剩余步数
        cell_time_step = cell_key_to_explore_from[0]
        remaining_steps = (H - 1) - cell_time_step
        
        # 收集经验数据
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []
        
        for explore_step in range(remaining_steps):
            if len(current_actions) >= H - 1:
                break
            
            current_time_step = len(current_actions)
            
            # 准备SAC输入状态
            # 在执行动作之前，构建状态历史（不包含即将产生的新状态）
            # current_states 包含到目前为止的所有状态（包括从cell恢复的状态和新探索的状态）
            state_input = append_state(current_states, H-1).squeeze().numpy()
            
            # 使用SAC选择动作
            action = sac_agent.select_action(state_input)
            
            # 执行动作
            current_actions.append(action)
            env.step(0, torch.tensor([action]))
            
            new_state_tensor = env.state.clone()
            current_states.append(new_state_tensor)  # 添加新状态到历史中
            
            # 计算奖励
            current_reward = calculate_submodular_reward(current_states, env)
            
            # 计算即时奖励（相对于上一步的改进）
            if len(current_states) > 1:
                prev_reward = calculate_submodular_reward(current_states[:-1], env)
                immediate_reward = current_reward - prev_reward
            else:
                immediate_reward = current_reward
            
            # 准备下一状态
            # 检查是否是最后一步，如果是，下一状态就是当前状态
            if len(current_actions) >= H - 1:
                next_state_input = state_input  # 最后一步，没有真正的下一状态
                done = True
            else:
                # 不是最后一步，下一状态包含新添加的状态
                next_state_input = append_state(current_states, H-1).squeeze().numpy()
                done = False
            
            # 存储经验
            episode_states.append(state_input)
            episode_actions.append(action)
            episode_rewards.append(immediate_reward)
            episode_next_states.append(next_state_input)
            episode_dones.append(done)
            
            # 添加到replay buffer
            replay_buffer.push(state_input, action, immediate_reward, 
                             next_state_input, done)
            
            # 更新存档库
            new_state_id = new_state_tensor.item()
            total_actions_taken = len(current_actions)
            new_cell_key = (total_actions_taken, new_state_id)
            
            new_reward = calculate_submodular_reward(current_states, env)
            
            if new_cell_key not in archive or new_reward > archive[new_cell_key]['reward']:
                archive[new_cell_key] = {
                    'reward': new_reward,
                    'states': current_states[:],
                    'actions': current_actions[:],
                    'times_selected': 0
                }
        
        # 3.4 更新SAC网络
        if len(replay_buffer) > 256:
            for _ in range(10):  # 每次探索后进行多次更新
                sac_agent.update(replay_buffer, batch_size=256)
        
        # 定期输出信息
        if step % 100 == 0:
            print(f"当前存档库大小: {len(archive)}")
            _best_trajectory_data = max(archive.values(), key=lambda x: x['reward'])
            _max_reward = _best_trajectory_data['reward']
            print(f"找到的最佳奖励值: {_max_reward}")
            print(f"对应的轨迹长度: {len(_best_trajectory_data['states'])}")
            
            # 策略评估
            evaluate_sac_policy(sac_agent, env, params, num_episodes=5)
                
        pbar.update(1)
    pbar.close()

    # 4. 探索结束
    print("\n--- 探索完成 ---")
    if not archive:
        print("错误：存档库为空！")
        return None, None

    best_trajectory_data = max(archive.values(), key=lambda x: x['reward'])
    max_reward = best_trajectory_data['reward']
    print(f"找到的最佳奖励值: {max_reward}")
    print(f"对应的轨迹长度: {len(best_trajectory_data['states'])}")

    # 5. 保存存档库
    archive_filename = "sac_go_explore_archive.pkl"
    with open(archive_filename, "wb") as f:
        pickle.dump(archive, f)
    print(f"完整的时空细胞存档库已保存至: {archive_filename}")
    
    return best_trajectory_data, sac_agent

def evaluate_sac_policy(sac_agent, env, params, num_episodes=10):
    """
    评估训练好的SAC策略
    """
    print("--- 评估训练好的SAC策略 ---")
    
    H = params["env"]["horizon"]
    total_rewards = []
    
    for episode in range(num_episodes):
        env.initialize(params["env"]["initial"])
        mat_state = [env.state.clone()]
        mat_action = []
        
        for h_iter in range(H-1):
            # 准备SAC输入状态 - 使用状态历史
            state_input = append_state(mat_state, H-1).squeeze().numpy()
            
            # 使用SAC选择动作（确定性策略用于评估）
            with torch.no_grad():
                action_probs = sac_agent.actor(torch.FloatTensor(state_input).unsqueeze(0).to(sac_agent.device))
                action = torch.argmax(action_probs, dim=-1).item()
            
            mat_action.append(action)
            env.step(h_iter, torch.tensor([action]))
            mat_state.append(env.state.clone())
        
        # 计算该轮次的总奖励
        episode_reward = calculate_submodular_reward(mat_state, env)
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"平均奖励: {avg_reward}, 最大奖励: {max(total_rewards)}, 最小奖励: {min(total_rewards)}")
    
    return total_rewards, avg_reward

# --- 主执行函数 ---
def main():
    """
    主函数示例
    """
    # 这里需要根据实际环境和参数进行配置
    # env = GridWorld()
    # params = {
    #     "env": {
    #         "horizon": 10,
    #         "initial": "some_initial_config",
    #         "disc_size": "small"
    #     },
    #     "alg": {
    #         "lr": 3e-4,
    #         "type": "SAC"
    #     }
    # }
    # 
    # best_trajectory, trained_agent = run_sac_with_go_explore(env, params)
    pass

if __name__ == "__main__":
    main()
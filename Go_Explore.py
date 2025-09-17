import os
import random
import dill as pickle
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from subrl.utils.environment import GridWorld
from subrl.utils.network import append_state
from subrl.utils.network import policy as agent_net
from subrl.utils.visualization import Visu
from subpo import calculate_submodular_reward, compute_subpo_advantages

def select_cell_from_archive(archive, current_step=0, total_steps=1):
    """
    Select a cell from the archive for exploration.
    - Before 4/5 of total steps: Cells with fewer steps are prioritized using probability weights (Weight = 1/(time_step + 1))
    - After 4/5 of total steps: Use uniform sampling
    
    Args:
        archive: The cell archive dictionary
        current_step: Current exploration step
        total_steps: Total exploration steps
    """
    if not archive:
        return None, None

    cell_keys = list(archive.keys())
    
    # 检查是否已达到总步数的4/5
    progress_ratio = current_step / total_steps if total_steps > 0 else 0
    use_uniform_sampling = progress_ratio >= 0.8  # 4/5 = 0.8
    
    if use_uniform_sampling:
        # 使用均匀采样
        normalized_weights = [1.0 / len(cell_keys) for _ in cell_keys]
    else:
        # 使用步数加权采样：计算每个cell的权重：1/(步数+1)
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
            # 如果所有权重为0，使用均匀分布
            normalized_weights = [1.0 / len(cell_keys) for _ in cell_keys]
    
    # 根据权重随机选择cell
    import numpy as np
    selected_index = np.random.choice(len(cell_keys), p=normalized_weights)
    selected_cell_key = cell_keys[selected_index]
    
    return selected_cell_key, archive[selected_cell_key]

def sample_excellent_trajectories(filepath: str, 
                                  method='top_n', 
                                  n=10, 
                                  p=0.1, 
                                  threshold=0):
    """
        Load data from the Go-Explore archive and sample high-quality trajectories based on the specified method.

        Args:
            filepath (str): Path to the .pkl archive file.
            method (str): Sampling method. Options are 'top_n', 'top_p', or 'threshold'.
            n (int): Number of trajectories to sample for the 'top_n' method.
            p (float): Percentage of top trajectories to sample for the 'top_p' method (e.g., 0.1 means top 10%).
            threshold (float): Minimum reward threshold for the 'threshold' method.
        
        Returns:
            list: A list of trajectory dictionaries with high rewards, sorted in descending order of reward.
                  Returns an empty list if the file does not exist or the archive is empty.
    """
    # 1. Check if the file exists and load the data
    if not os.path.exists(filepath):
        print(f"Error: Archive file not found '{filepath}'")
        return []
    
    try:
        with open(filepath, "rb") as f:
            archive = pickle.load(f)
        if not archive:
            print("警告：存檔庫為空。")
            return []
    except Exception as e:
        print(f"讀取文件時出錯: {e}")
        return []

    # 2. 提取所有軌跡數據並按獎勵排序
    # archive.values() 返回的是包含 reward, states, actions 等信息的字典
    all_trajectories_data = list(archive.values())
    
    # 按 'reward' 鍵從高到低排序
    all_trajectories_data.sort(key=lambda x: x['reward'], reverse=True)

    # 3. 根據指定方法進行採樣
    sampled_trajectories = []
    if method == 'top_n':
        # 取獎勵最高的前 N 條
        num_to_sample = min(n, len(all_trajectories_data))
        sampled_trajectories = all_trajectories_data[:num_to_sample]
        print(f"方法: Top-N。從 {len(all_trajectories_data)} 條軌跡中篩選出最好的 {len(sampled_trajectories)} 條。")

    elif method == 'top_p':
        # 取獎勵最高的前 P%
        if not (0 < p <= 1):
            print("錯誤：百分比 'p' 必須在 (0, 1] 之間。")
            return []
        num_to_sample = int(len(all_trajectories_data) * p)
        sampled_trajectories = all_trajectories_data[:num_to_sample]
        print(f"方法: Top-P。從 {len(all_trajectories_data)} 條軌跡中篩選出最好的前 {p*100:.1f}% ({len(sampled_trajectories)} 條)。")

    elif method == 'threshold':
        # 取獎勵高於指定門檻的所有軌跡
        sampled_trajectories = [data for data in all_trajectories_data if data['reward'] >= threshold]
        print(f"方法: Threshold。從 {len(all_trajectories_data)} 條軌跡中篩選出 {len(sampled_trajectories)} 條獎勵不低於 {threshold} 的軌跡。")
        
    else:
        print(f"錯誤：未知的採樣方法 '{method}'。請使用 'top_n', 'top_p', 或 'threshold'。")

    return sampled_trajectories

# --- Core parameter setting ---
EXPLORATION_STEPS = 50000  # The total number of exploration steps can be adjusted as needed, and the larger the value, the more thorough the exploration 

# --- Go-Explore main  ---
def run_srl_with_go_explore(env=GridWorld, params=None):
    """
    執行 SRL 結合 Go-Explore 的訓練（在SRL基礎上增加Go-Explore的探索策略）
    """
    print("--- SRL with Go-Explore Training 開始 ---")
    
    # 1. 初始化存檔庫 (Archive)
    # ---【核心修改】：將細胞 Key 定義為 (時間, 狀態ID) ---
    # 結構: { (time, state_id): {'reward': float, 'states': list, 'actions': list, 'times_selected': int} }
    archive = {}
    
    # 2. 初始化策略網路和優化器
    H = params["env"]["horizon"]
    MAX_Ret = 2*(H+1)
    if params["env"]["disc_size"] == "large":
        MAX_Ret = 3*(H+2)
    
    # 根據算法類型初始化策略網路
    if params["alg"]["type"]=="M" or params["alg"]["type"]=="SRL":
        agent = agent_net(2, env.action_dim)  # (state, time) input
    else:
        agent = agent_net(H-1, env.action_dim)
    
    optim = torch.optim.Adam(agent.parameters(), lr=params["alg"]["lr"])
    
    env.common_params["batch_size"]=1
    env.initialize()
    initial_state_tensor = env.state.clone()
    initial_state_id = initial_state_tensor.item()
    
    # ---【核心修改】：初始細胞現在是 (時間=0, 狀態) ---
    initial_cell_key = (0, initial_state_id)
    
    initial_reward = calculate_submodular_reward([initial_state_tensor], env)
    
    archive[initial_cell_key] = {
        'reward': initial_reward,
        'states': [initial_state_tensor],
        'actions': [],
        'times_selected': 0
    }
    print(f"初始時空細胞加入存檔庫: Cell {initial_cell_key}, Reward: {initial_reward}")
    
    # 3. 執行 N 次探索循環
    pbar = tqdm(total=EXPLORATION_STEPS, desc="Go-Exploring (Spacetime)")
    for step in range(EXPLORATION_STEPS):
        # 3.1 選擇細胞 (传递当前步数和总步数)
        cell_key_to_explore_from, selected_cell_data = select_cell_from_archive(
            archive, current_step=step, total_steps=EXPLORATION_STEPS)
        
        if selected_cell_data is None:
            print("錯誤：存檔庫為空，無法繼續探索。")
            break
            
        archive[cell_key_to_explore_from]['times_selected'] += 1

        # 3.2 前往 (Go To) 該細胞狀態
        env.initialize()
        for action in selected_cell_data['actions']:
            env.step(0, torch.tensor([action]))

        # 3.3 使用策略網路執行剩餘全部步數
        current_states = selected_cell_data['states'][:]
        current_actions = selected_cell_data['actions'][:]
        
        # 為策略網路訓練收集數據
        mat_action = []
        mat_state = []
        mat_return = []
        marginal_return = []
        list_batch_state = []
        
        # 添加已有狀態到軌跡中
        mat_state.extend(current_states)
        init_state = current_states[0] if current_states else env.state.clone()
        
        # 為完整軌迹訓練準備：為cell中已有的狀態也構建batch_state
        for i in range(len(selected_cell_data['actions'])):
            if params["alg"]["type"]=="M" or params["alg"]["type"]=="SRL":
                # 對於已有的動作，使用執行動作前的狀態（即第i個狀態）
                cell_batch_state = selected_cell_data['states'][i].reshape(-1, 1).float()
                # 添加對應的時間索引（第i步動作對應時間索引i）
                cell_batch_state = torch.cat(
                    [cell_batch_state, i*torch.ones_like(cell_batch_state)], 1)
            else:
                # 構建到第i步的狀態序列（用於決策第i個動作）
                states_up_to_i = selected_cell_data['states'][:i+1]  # 包含初始狀態到第i個狀態
                cell_batch_state = append_state(states_up_to_i, H-1)
            list_batch_state.append(cell_batch_state)
        
        # 將cell中已有的動作也加入mat_action（用於完整軌迹訓練）
        mat_action.extend(selected_cell_data['actions'])
        
        # 計算cell已有部分的邊際獎勵
        for i in range(len(selected_cell_data['actions'])):
            states_up_to_i = selected_cell_data['states'][:i+2]  # 從初始狀態到第i+1個狀態
            reward_up_to_i = calculate_submodular_reward(states_up_to_i, env)
            mat_return.append(reward_up_to_i)
            
            if i == 0:
                marginal_return.append(reward_up_to_i)
            else:
                marginal_return.append(reward_up_to_i - mat_return[i-1])
        
        # 計算剩餘步數：直接使用cell key中的時間步信息
        cell_time_step = cell_key_to_explore_from[0]  # cell key的第一個元素是時間步（已執行的動作數）
        remaining_steps = (H - 1) - cell_time_step  # 剩餘可執行的動作步數
        
        for explore_step in range(remaining_steps):
            # 雙重檢查：確保不超過總horizon
            if len(current_actions) >= H - 1:
                break
            
            # 當前時間步 = 已執行的動作總數（包括cell記錄的和新執行的）
            current_time_step = len(current_actions)
            
            # 準備策略網路的輸入
            if params["alg"]["type"]=="M" or params["alg"]["type"]=="SRL":
                batch_state = env.state.reshape(-1, 1).float()
                # 添加時間索引到狀態
                batch_state = torch.cat(
                    [batch_state, current_time_step*torch.ones_like(batch_state)], 1)
            else:
                batch_state = append_state(mat_state, H-1)
            
            # 使用策略網路選擇動作
            action_prob = agent(batch_state)
            policy_dist = Categorical(action_prob)
            action = policy_dist.sample().item()
            
            mat_action.append(action)
            env.step(0, torch.tensor([action]))
            
            new_state_tensor = env.state.clone()
            
            current_states.append(new_state_tensor)
            current_actions.append(action)
            mat_state.append(new_state_tensor)
            
            # 計算獎勵和邊際獎勵
            current_reward = calculate_submodular_reward(current_states, env)
            mat_return.append(current_reward)
            
            # 計算邊際獎勵（相對於上一步的完整軌迹）
            if len(mat_return) == 1:
                marginal_return.append(mat_return[-1])
            else:
                marginal_return.append(mat_return[-1] - mat_return[-2])
            
            list_batch_state.append(batch_state)
            
            # ---【核心修改】：更新存檔庫時使用 (總動作數, 狀態) 作為 Key ---
            new_state_id = new_state_tensor.item()
            # 總動作數 = 當前已執行的動作總數
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
        
        # 3.4 訓練策略網路（如果有收集到新的經驗）
        if mat_action and list_batch_state:
            # 計算策略梯度
            states_visited = torch.vstack(list_batch_state).float()
            policy_dist = Categorical(agent(states_visited))
            log_prob = policy_dist.log_prob(torch.tensor(mat_action))
            batch_return = torch.tensor(marginal_return).float() / MAX_Ret

            # 計算損失函數
            J_obj = -1*(torch.mean(log_prob*batch_return) + params["alg"]["ent_coef"] *
                        policy_dist.entropy().mean()/(step+1))
            
            # 反向傳播和參數更新
            optim.zero_grad()
            J_obj.backward()
            optim.step()
        
        if step % 100 == 0:
            # print(f"探索步骤: {step / EXPLORATION_STEPS * 100:.2f}%")
            print(f"當前存檔庫大小: {len(archive)}")
            _best_trajectory_data = max(archive.values(), key=lambda x: x['reward'])
            _max_reward = _best_trajectory_data['reward']
            print(f"找到的最佳獎勵值: {_max_reward}")
            print(f"對應的軌跡長度: {len(_best_trajectory_data['states'])}")
            # if mat_action and list_batch_state:
            #     print(f"策略熵: {policy_dist.entropy().mean().detach()}")
            
            # 策略評估：每隔100轮评估当前策略性能
            # print(f"\n--- 第 {step} 步策略評估 ---")
            evaluate_trained_policy(agent, env, params, num_episodes=5)
            # print(f"當前策略平均獎勵: {eval_avg:.4f}")
            # print("--- 策略評估完成 ---\n")
                
        pbar.update(1)
    pbar.close()

    # 4. 探索結束
    print("\n--- 探索完成 ---")
    if not archive:
        print("錯誤：存檔庫為空！")
        return None, None

    best_trajectory_data = max(archive.values(), key=lambda x: x['reward'])
    max_reward = best_trajectory_data['reward']
    print(f"找到的最佳獎勵值: {max_reward}")
    print(f"對應的軌跡長度: {len(best_trajectory_data['states'])}")

    # 5. 保存存檔庫
    archive_filename = "srl_go_explore_archive.pkl"
    with open(archive_filename, "wb") as f:
        pickle.dump(archive, f)
    print(f"完整的時空細胞存檔庫已保存至: {archive_filename}")
    
    return best_trajectory_data, agent

# --- 運行 SRL with Go-Explore ---
# best_found_trajectory, trained_agent = run_srl_with_go_explore(env, params)

def evaluate_trained_policy(agent, env, params, num_episodes=10):
    """
    評估訓練好的策略網路
    """
    print("--- 評估訓練好的策略網路 ---")
    
    H = params["env"]["horizon"]
    total_rewards = []
    
    for episode in range(num_episodes):
        env.initialize()
        mat_state = [env.state.clone()]
        mat_action = []
        
        for h_iter in range(H-1):
            if params["alg"]["type"]=="M" or params["alg"]["type"]=="SRL":
                batch_state = mat_state[-1].reshape(-1, 1).float()
                batch_state = torch.cat(
                    [batch_state, h_iter*torch.ones_like(batch_state)], 1)
            else:
                batch_state = append_state(mat_state, H-1)
            
            with torch.no_grad():
                action_prob = agent(batch_state)
                # 使用概率選择進行評估
                policy_dist = Categorical(action_prob)
                action = policy_dist.sample().item()
            
            mat_action.append(action)
            env.step(h_iter, torch.tensor([action]))
            mat_state.append(env.state.clone())
        
        # 計算該輪次的總獎勵
        episode_reward = calculate_submodular_reward(mat_state, env)
        total_rewards.append(episode_reward)
        
        # print(f"Episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"平均獎勵: {avg_reward},最大獎勵: {max(total_rewards)},最小獎勵: {min(total_rewards)}")
    # print(f"最大獎勵: {max(total_rewards)}")
    # print(f"最小獎勵: {min(total_rewards)}")
    
    return total_rewards, avg_reward
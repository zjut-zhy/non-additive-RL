import os
import random
import dill as pickle
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from environment import GridWorld
from subrl.utils.network import append_state
from subrl.utils.network import policy as agent_net
from visualization import Visu
from subpo import calculate_submodular_reward, compute_subpo_advantages
from Go_Explore import run_srl_with_go_explore, evaluate_trained_policy
workspace = "NM"


params = {
    "env": {
        "start": 1,
        "step_size": 0.1,
        "shape": {"x": 11, "y": 18},
        "horizon": 80,
        "node_weight": "constant",
        "disc_size": "small",
        "n_players": 3,
        "Cx_lengthscale": 2,
        "Cx_noise": 0.001,
        "Fx_lengthscale": 1,
        "Fx_noise": 0.001,
        "Cx_beta": 1.5,
        "Fx_beta": 1.5,
        "generate": False,
        "env_file_name": 'env_data.pkl',
        "cov_module": 'Matern',
        "stochasticity": 0.0,
        "domains": "two_room_2",
        "num": 1,  # 替代原来的args.env
        "initial": 80
    },
    "alg": {
        "gamma": 1,
        "type": "NM",
        "ent_coef": 0.0,
        "epochs": 140,
        "lr": 0.02
    },
    "common": {
        "a": 1,
        "subgrad": "greedy",
        "grad": "pytorch",
        "algo": "both",
        "init": "deterministic",
        "batch_size": 1
    },
    "visu": {
        "wb": "disabled",
        "a": 1
    }
}
env_load_path = workspace + \
    "/environments/" + params["env"]["node_weight"]+ "/env_1" 

params['env']['num'] = 1
# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="code-" + params["env"]["node_weight"],
#     mode=params["visu"]["wb"],
#     config=params
# )

epochs = params["alg"]["epochs"]

H = params["env"]["horizon"]
MAX_Ret = 2*(H+1)
if params["env"]["disc_size"] == "large":
    MAX_Ret = 3*(H+2)
    
env = GridWorld(
    env_params=params["env"], common_params=params["common"], visu_params=params["visu"], env_file_path=env_load_path)
node_size = params["env"]["shape"]['x']*params["env"]["shape"]['y']
# TransitionMatrix = torch.zeros(node_size, node_size)

if params["env"]["node_weight"] == "entropy" or params["env"]["node_weight"] == "steiner_covering" or params["env"]["node_weight"] == "GP": 
    a_file = open(env_load_path +".pkl", "rb")
    data = pickle.load(a_file)
    a_file.close()

if params["env"]["node_weight"] == "entropy":
    env.cov = data
if params["env"]["node_weight"] == "steiner_covering":
    env.items_loc = data
if params["env"]["node_weight"] == "GP":
    env.weight = data

visu = Visu(env_params=params["env"])
# plt, fig = visu.stiener_grid( items_loc=env.items_loc, init=34)
# wandb.log({"chart": wandb.Image(fig)})
# plt.close()
# Hori_TransitionMatrix = torch.zeros(node_size*H, node_size*H)
# for node in env.horizon_transition_graph.nodes:
#     connected_edges = env.horizon_transition_graph.edges(node)
#     for u, v in connected_edges:
#         Hori_TransitionMatrix[u[0]*node_size+u[1], v[0]*node_size + v[1]] = 1.0
env.get_horizon_transition_matrix()

# --- 运行 SRL with Go-Explore ---
print("开始运行 SRL with Go-Explore...")
best_found_trajectory, trained_agent = run_srl_with_go_explore(env, params)

if best_found_trajectory is not None:
    print(f"\n训练完成！")
    print(f"最佳轨迹奖励: {best_found_trajectory['reward']}")
    print(f"最佳轨迹长度: {len(best_found_trajectory['states'])}")
    print("Archive文件已保存，训练完成。")
else:
    print("训练失败：未找到有效轨迹")



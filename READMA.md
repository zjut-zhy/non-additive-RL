# 非可加性强化学习实验

该项目是探索和实现各种在线和离线强化学习算法的集合。实验在自定义的 `GridWorld` 环境（“one_room” 和 “two_rooms”）中进行。

项目重点研究了从预先收集的数据集中进行离线学习，特别是使用保守Q学习（CQL）和行为克隆（BC）等算法。

## 特性

- **已实现的强化学习算法**:
  - Advantage Actor-Critic (A2C)
  - Proximal Policy Optimization (PPO)
  - Deep Q-Network (DQN)
  - Soft Actor-Critic (SAC)
  - Conservative Q-Learning (CQL) (离线)
  - Behavior Cloning (BC) (离线)
- **环境**:
  - 自定义 `GridWorld` 环境: `one_room` 和 `two_rooms`。
- **离线学习**:
  - 支持从存储的轨迹数据中采样高质量的专家数据。
  - 使用采样数据进行离线智能体训练。
- **核心库**:
  - `subrl` 模块提供了环境、网络结构和可视化等核心工具。

## 安装

1.  克隆仓库。
2.  安装所需的依赖项。建议使用 `pip` 和 [subrl/requirements.txt](subrl/requirements.txt) 文件来创建虚拟环境。

    ```sh
    pip install -r subrl/requirements.txt
    ```

## 如何使用

项目中的每个主要算法都在其自己的 Jupyter Notebook (`.ipynb`) 文件中实现和演示。您可以直接打开并运行这些 notebook 来查看算法的训练和评估过程。

- **在线算法**:
  - [A2C_one_room.ipynb](A2C_one_room.ipynb) 和 [A2C_two_rooms.ipynb](A2C_two_rooms.ipynb)
  - [PPO_one_room.ipynb](PPO_one_room.ipynb) 和 [PPO_two_rooms.ipynb](PPO_two_rooms.ipynb)
  - [DQN_one_room.ipynb](DQN_one_room.ipynb) 和 [DQN_two_rooms.ipynb](DQN_two_rooms.ipynb)
  - [SAC_one_room.ipynb](SAC_one_room.ipynb) 和 [SAC_two_rooms.ipynb](SAC_two_rooms.ipynb)

- **离线算法**:
  - [BC_one_room.ipynb](BC_one_room.ipynb) 和 [BC_two_rooms.ipynb](BC_two_rooms.ipynb)
  - [CQL_one_room.ipynb](CQL_one_room.ipynb) 和 [CQL_two_rooms.ipynb](CQL_two_rooms.ipynb)

- **核心实验**:
  - [main_NM_subpo_go_explore_3_2.ipynb](main_NM_subpo_go_explore_3_2.ipynb) 包含使用 Go-Explore 收集的数据进行离线学习和模仿学习的详细流程。

## 数据

项目使用 `.pkl` 文件（例如 [go_explore_archive_spacetime_.pkl](go_explore_archive_spacetime_.pkl)）来存储收集的轨迹数据。这些数据用于离线算法的训练。

## 贡献

欢迎通过 Pull Request 或 Issues 为该项目做出贡献。

## 许可证

该项目采用 MIT 许可证。详情请见 [subrl/LICENSE](subrl/LICENSE) 文件。
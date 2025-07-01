import numpy as np
def calculate_submodular_reward(trajectory, env):
    """
    The wrapper function is used to calculate the submodular reward for a given trajectory.
    Directly call the env.weighted_traj_deturn method that already exists in your environment.
    """
    if not trajectory:
        return 0
    formatted_traj = [state.reshape(1, -1) for state in trajectory]

    return env.weighted_traj_return(formatted_traj).item()


def compute_subpo_advantages(trajectory_states, env, baseline_mode='mean'):
    """
    Calculate the correct SUBPO advantage function for a single trajectory.
    Advantage A (i)=(∑ _ {j=i} ^ {H-1} F (s_ {j+1} | τ _ {0: j}) - baseline
    """
    H = len(trajectory_states)
    advantages = np.zeros(H)
    marginal_gains = np.zeros(H)

    # 1. Firstly, calculate the marginal gain for each step F(s_{j+1}|τ_{0:j})
    reward_so_far = calculate_submodular_reward([], env)  # The reward for an empty trajectory is 0
    for j in range(H):
        # Trajectory τ _ {0: j+1} is the first j+1 element of the trajectory
        sub_trajectory = trajectory_states[:j + 1]
        reward_now = calculate_submodular_reward(sub_trajectory, env)

        # Marginal gain = F(τ_{0:j+1}) - F(τ_{0:j})
        marginal_gains[j] = reward_now - reward_so_far
        reward_so_far = reward_now

    # 2. Calculate the advantage value of each step (sum of future marginal gains)
    for i in range(H):
        future_marginal_gains_sum = np.sum(marginal_gains[i:])
        advantages[i] = future_marginal_gains_sum

    # 3. (Optional but recommended) Subtract a baseline to reduce variance
    if baseline_mode == 'mean' and H > 1:
        advantages -= np.mean(advantages)

    return advantages.tolist()
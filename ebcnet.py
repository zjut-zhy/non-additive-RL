# 策略网络训练模块
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TemporalStateEncoder(nn.Module):
    def __init__(self, num_states=98, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim)
        self.hidden_dim = hidden_dim  # 保存 hidden_dim 以便在 forward 中使用

    def forward(self, state_seq_batch):
        """
        MODIFIED: This forward method now correctly handles a batch of sequences.
        :param state_seq_batch: A list of sequences, e.g., [[-1, 10], [3, 4, 5]]
        :return: A tensor of embeddings for the batch, shape [B, hidden_dim]
        """
        #如果是张量数据，并行
        if isinstance(state_seq_batch, torch.Tensor):
            # 张量输入：并行处理
            batch_tensor = state_seq_batch.to(self.embedding.weight.device)
            
            # 获取embeddings：[B, T, embed_dim]
            embeddings = self.embedding(batch_tensor)
            # 转换为 [T, B, embed_dim] 格式
            embeddings = embeddings.transpose(0, 1)
            
            # 通过LSTM：[T, B, embed_dim] -> h_n: [1, B, hidden_dim]
            _, (h_n, _) = self.lstm(embeddings)
            
            # 返回最后一层的hidden state：[B, hidden_dim]
            return h_n.squeeze(0)
        
        batch_embeddings = []
        # 遍历批次中的每一条序列
        for state_seq in state_seq_batch:
            # --- 内部逻辑处理单条序列 ---
            # indices = [i for i in state_seq if i >= 0]  # 过滤掉填充值
            indices = state_seq

            # 转换为张量并获取嵌入
            input_tensor = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)
            input_emb = self.embedding(input_tensor).unsqueeze(1)  # [T, 1, D]

            # 通过 LSTM
            _, (h_n, _) = self.lstm(input_emb)

            # 将这条序列的结果添加到批次列表中
            batch_embeddings.append(h_n.squeeze(0).squeeze(0))

        # 将所有序列的结果堆叠成一个批次张量并返回
        return torch.stack(batch_embeddings)

class EmbeddingBehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device='cpu'):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, batch_states, batch_actions, total_loss):
        logits = self.policy(batch_states)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, batch_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        return total_loss

    def predict_action_batch_sample(self, state_batch):
        if state_batch.dim() == 1:
            state_batch = state_batch.unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_batch)
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).tolist()
        return actions, probs.tolist()


class EndToEndModel(nn.Module):
    def __init__(self, num_states, embed_dim, encoder_hidden_dim, policy_hidden_dim, action_dim):
        super().__init__()
        # 在内部创建编码器和策略网络实例
        self.encoder = TemporalStateEncoder(num_states, embed_dim, encoder_hidden_dim)
        self.policy = PolicyNetwork(encoder_hidden_dim, policy_hidden_dim, action_dim)

    def forward(self, raw_state_sequences):
        """
        定义从原始状态序列到最终动作logits的完整流程。
        :param raw_state_sequences: 一个批次的原始历史轨迹, e.g., [[-1, 10, 20], [3, 4, -1]]
        :return: 动作的 logits, shape [B, action_dim]
        """
        # 1. 首先通过编码器，将原始序列转换为状态嵌入向量
        state_embeddings = self.encoder(raw_state_sequences)

        # 2. 然后将嵌入向量输入策略网络，得到最终的动作 logits
        action_logits = self.policy(state_embeddings)

        return action_logits


class EndToEndBehaviorCloning:
    def __init__(self, num_states, embed_dim, encoder_hidden_dim, policy_hidden_dim, action_dim, lr, device='cpu'):
        self.device = device

        # 1. 初始化完整的端到端模型
        self.model = EndToEndModel(
            num_states, embed_dim, encoder_hidden_dim,
            policy_hidden_dim, action_dim
        ).to(self.device)

        # 2. 关键：优化器管理组合模型的所有参数
        #    这样，梯度就能同时更新编码器和策略网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, batch_sequences, batch_actions):
        """
        执行一步训练。
        :param batch_sequences: 一个批次的原始历史轨迹
        :param batch_actions: 对应的专家动作
        """
        self.model.train()  # 设置为训练模式

        # 将动作数据移动到正确的设备
        batch_actions = batch_actions.to(self.device)

        # 直接调用组合模型进行前向传播
        logits = self.model(batch_sequences)

        # 计算损失
        loss = self.criterion(logits, batch_actions)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()  # 梯度会在这里流遍整个 EndToEndModel (包括encoder和policy)
        self.optimizer.step()

        return loss.item()

    def predict(self, sequence):
        """对单个序列进行预测"""
        self.model.eval()
        with torch.no_grad():
            # 需要在批次维度上增加一维
            logits = self.model([sequence])
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()
        return action

    def predicts(self, sequence):
        """对多个序列进行预测"""
        self.model.eval()
        with torch.no_grad():
            # 需要在批次维度上增加一维
            logits = self.model(sequence)
            probs = torch.softmax(logits, dim=1)
            actions = torch.argmax(probs, dim=1)
        return actions


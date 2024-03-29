# DDQN_Pytorch
> 1 依赖库：Python -> 3.6、Pytorch -> 1.6、numpy、matplotlib、gym
>
> 2 训练：配置好环境后，直接运行train.py即可

当然，让我们更详细地分析这两个类中的每个函数：

### `DeepQNetwork` 类

1. **构造函数 `__init__`**
   - `alpha`: 学习率，控制模型在学习过程中的步长。
   - `state_dim`: 状态维度，即神经网络输入层的大小，对应于环境状态的维度。
   - `action_dim`: 动作维度，即神经网络输出层的大小，对应于可能的动作数量。
   - `fc1_dim`, `fc2_dim`: 第一和第二个全连接层的神经元数量。
   - 此函数初始化网络的结构，包括两个全连接层和一个输出层，以及为网络训练定义的优化器（Adam）。

2. **前向传播函数 `forward`**
   - `state`: 输入的状态。
   - 该函数定义了网络如何处理输入的状态，并输出每个动作的预期 Q 值。它通过两个全连接层和激活函数（ReLU）来处理输入。

3. **保存模型 `save_checkpoint`**
   - `checkpoint_file`: 保存模型权重的文件路径。
   - 此函数将当前网络的状态（权重等）保存到指定的文件中。

4. **加载模型 `load_checkpoint`**
   - `checkpoint_file`: 从中加载模型权重的文件路径。
   - 此函数从指定文件中加载网络状态。

### `DDQN` 类

1. **构造函数 `__init__`**
   - 这个函数初始化 DDQN 算法所需的各种参数和对象。
   - 参数包括学习率 (`alpha`)，状态和动作维度（`state_dim`, `action_dim`），两个全连接层的维度（`fc1_dim`, `fc2_dim`），模型检查点存储路径（`ckpt_dir`），折扣因子（`gamma`），目标网络软更新参数（`tau`），探索率（`epsilon`）及其衰减相关参数（`eps_end`, `eps_dec`），经验回放的最大大小（`max_size`）和批处理大小（`batch_size`）。
   - 该函数创建了两个 `DeepQNetwork` 实例（评估网络和目标网络），一个经验回放缓冲区，以及一个动作空间列表。

2. **更新网络参数 `update_network_parameters`**

- 这个方法用于软更新目标网络 (`q_target`) 的参数。它使用 `tau` 参数来控制目标网络参数的更新速率，使其更平滑地跟踪评估网络 (`q_eval`) 的参数。初始调用会强制同步两个网络的参数。

3. **选择动作 `choose_action`**

- `observation`: 当前环境状态。
- 该方法根据当前状态和探索-利用策略选择一个动作。当随机数小于 `epsilon`（探索率）时，它会随机选择一个动作，否则会根据评估网络 (`q_eval`) 选择最佳动作。

4. **递减探索率 `decrement_epsilon`**

- 这个方法逐渐减少 `epsilon` 值，以随着训练的进行减少随机探索的次数，逐渐过渡到更多地利用模型的预测。

5. **学习方法 `learn`**

- 这是 DDQN 算法的核心，用于基于经验回放缓冲区的数据更新网络参数。
- 它首先检查内存是否准备好进行学习（是否积累了足够的经验）。然后，它从经验回放缓冲区中抽取一个批量的经验（状态、动作、奖励等），并使用这些数据来计算损失和更新评估网络 (`q_eval`)。该方法还包括更新目标网络的参数和递减探索率。

6. **保存模型 `save_models`**

- `episode`: 当前训练周期（通常是迭代次数或游戏回合数）。
- 该方法保存评估网络和目标网络的状态到指定的文件路径。

7. **加载模型 `load_models`**

- `episode`: 指定训练周期的模型状态要被加载。
- 该方法从文件中加载评估网络和目标网络的状态。

这些函数共同实现了深度双 Q 网络（DDQN）算法，一个在强化学习领域广泛使用的算法，它适用于具有离散动作空间的任务，如视频游戏或机器人控制。代码结合了神经网络（用于近似 Q 函数）、经验回放（用于打破数据的相关性并重复使用过去的经验），以及双 Q 学习（用于减少目标 Q 值的过估计）。

同时为了比较DQN与DDQN，我在myDQN.py文件中实现了DQN网络，并且再trainDQN.py文件中对模型训练，保存到DQN文件夹下，output_images_DQN文件夹下存储输出的图片。

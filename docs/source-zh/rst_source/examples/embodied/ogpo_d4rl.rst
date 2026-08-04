在 D4RL Adroit Relocate 上运行 OGPO
=====================================

本示例将官方低维 OGPO Relocate 配方接入 RLinf，配置文件为
``examples/embodiment/config/d4rl_ogpo_relocate.yaml``。

算法一致性
----------

Actor 直接输入 39 维原始状态，用 4 层、每层 512 单元的 GELU MLP
预测展平后的 4 步动作；时间使用官方 32 维正弦嵌入。Critic 是 10 个
相互独立的 4 层 512 单元 GELU MLP，并使用 LayerNorm。

训练分两个阶段。Flow-matching BC 最多更新 50,000 次，当 SDE 成功率达到
0.45 时可提前结束；随后进行 500,000 个 primitive transition 的在线训练。
在线阶段收集满 10,000 个 primitive transition 后开始更新，每个 transition
等效执行 4 次 critic 更新和 1 次 actor 更新，并保持官方的“先 actor、更新
target actor，再 critic、更新 target critic”顺序。Replay 仅提交完整 episode
的 primitive transition，再从任意起点构造 4 步序列；跨 terminal 的动作和
Q loss 由 ``valid`` 屏蔽。成功 episode 在达到官方容量门槛后用于在线 BC
正则。

RLinf 默认通过 PyTorch/Ray 并行运行 32 个训练环境，因此墙钟顺序和随机样本
不会与官方单环境 JAX 代码逐 bit 相同；网络、损失、调度、replay
语义、优化器参数以及按 transition 计算的有效 UTD 比例与官方 Adroit 配方一致。

启动训练
--------

先安装 D4RL 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env d4rl

将转换后的专家数据放到仓库上级目录下的
``datasets/relocate-expert-minari-v2.hdf5``，或将 ``OGPO_RELOCATE_DATASET``
设为数据文件的绝对路径，然后运行：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh d4rl_ogpo_relocate

指定单张物理 GPU（例如 1 号卡）时，可以直接运行：

.. code-block:: bash

   OGPO_GPU_PLACEMENT=1 bash examples/embodiment/run_embodiment.sh d4rl_ogpo_relocate

``OGPO_GPU_PLACEMENT`` 表示 Ray 硬件 rank。


默认使用 TensorBoard。训练指标位于 ``train/``，ODE 评估位于 ``eval/``，
SDE 评估位于 ``eval_sde/``。重点关注 ``ogpo/bc_update_steps``、
``ogpo/online_env_steps``、actor 的 PPO/BC 指标、critic TD 指标以及成功缓冲区
容量和启用状态。

Checkpoint 与评估
-----------------

系统会在 BC 切换到在线阶段时、每 100,000 个在线 primitive transition、
以及训练结束时保存 checkpoint。checkpoint 包含当前/target 模型、两套
optimizer 和 scheduler、replay、未完成的序列窗口、成功缓冲区以及 OGPO
阶段计数器，路径为
``<log_path>/<experiment_name>/checkpoints/global_step_N``。

续训时把 ``runner.resume_dir`` 指向上述 ``global_step_N`` 目录。单独进行
ODE 和 SDE 评估时，同时设置相同的 ``runner.resume_dir`` 和
``runner.only_eval=true``；不要把 ``resume_dir`` 指向其 ``actor`` 子目录。

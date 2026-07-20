LIBERO-Safety
=============

RLinf 将 LIBERO-Safety 注册为 ``env_type: libero_safety``。默认
``libero_safety`` suite 聚合上游提供仿真初始状态的四个物理安全 suite：
affordance、human_safety、obstacle_avoidance 和
obstacle_avoidance_human，共 60 个带难度级别的任务、3000 个评估 trial。
``reasoning_safety`` 未包含在默认仿真集合中，因为上游没有提供它的 init state。

环境准备
--------

.. code-block:: bash

   LIBERO_SAFETY_PATH=/path/to/LIBERO-Safety \
      bash requirements/install.sh embodied --model openpi --env libero_safety

安装脚本会安装已克隆的 LIBERO-Safety 及其内置 robosuite，并自动下载所需
assets、解压到 ``LIBERO-Safety/libero/libero/assets``。RLinf 默认查找同级
目录；若源码位于其他位置，安装时设置 ``LIBERO_SAFETY_PATH``，运行时也可设置
``LIBERO_SAFETY_REPO_PATH`` 或 ``env.train.repo_path``。RLinf 会生成隔离的
路径配置，不使用上游 ``libero/config.yaml`` 中可能失效的绝对路径。

数据集仓库没有发布 LeRobot 所需的 ``v2.1`` tag，因此需要显式从
``main`` revision 下载到 ``HF_LEROBOT_HOME``：

.. code-block:: bash

   hf download LIBERO-Safety/libero_safety \
      --repo-type dataset \
      --revision main \
      --local-dir "$HF_LEROBOT_HOME/LIBERO-Safety/libero_safety"

配置预期从
``$HF_LEROBOT_HOME/LIBERO-Safety/libero_safety/norm_stats.json`` 读取归一化
统计；若下载的数据集中没有该文件，请将计算结果放到这一位置。开始训练前仍需
按实际情况填写模型 checkpoint 路径。

训练命令
--------

OpenPI SFT 与 PPO：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_safety_sft_openpi_pi05
   bash examples/embodiment/run_embodiment.sh libero_safety_ppo_openpi_pi05

RLT Stage 1 与 Stage 2：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_safety_rlt_stage1_sft_openpi_pi05
   bash examples/embodiment/run_embodiment.sh libero_safety_rlt_stage2_ac_mlp

Stage 2 的 ``rollout.rlt_feature_model.model_path`` 需要指向 Stage 1 actor
checkpoint。任务成功获得 ``reward_coef``；任意 BDDL 安全约束触发的每一步会
扣除 ``safety_cost_coef``。环境还会记录 ``safety_cost`` 和 episode 级
``safety_violation_once``。如需单独训练某个 suite，可覆盖
``task_suite_name``。

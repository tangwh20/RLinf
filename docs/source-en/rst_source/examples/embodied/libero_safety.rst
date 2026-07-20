LIBERO-Safety
=============

RLinf exposes LIBERO-Safety as ``env_type: libero_safety``. The default
``libero_safety`` task suite aggregates the four physical suites that ship
simulation init states: affordance, human safety, obstacle avoidance, and
obstacle avoidance with a moving human. This is 60 level-specific tasks and
3000 evaluation trials. ``reasoning_safety`` is not included because the
upstream repository does not ship simulator init states for it.

Setup
-----

.. code-block:: bash

   LIBERO_SAFETY_PATH=/path/to/LIBERO-Safety \
      bash requirements/install.sh embodied --model openpi --env libero_safety

The installer installs the cloned LIBERO-Safety repository and its vendored
robosuite, and automatically downloads and extracts the required assets into
``LIBERO-Safety/libero/libero/assets``. RLinf defaults to a sibling checkout;
set ``LIBERO_SAFETY_PATH`` during installation, or
``LIBERO_SAFETY_REPO_PATH`` / ``env.train.repo_path`` at runtime, when it lives
elsewhere. RLinf creates an isolated LIBERO path config, so the stale absolute
paths in the upstream ``libero/config.yaml`` are not used.

The dataset repository does not publish the ``v2.1`` tag expected by
LeRobot, so download its ``main`` revision explicitly into
``HF_LEROBOT_HOME``:

.. code-block:: bash

   hf download LIBERO-Safety/libero_safety \
      --repo-type dataset \
      --revision main \
      --local-dir "$HF_LEROBOT_HOME/LIBERO-Safety/libero_safety"

The configs expect the normalization statistics at
``$HF_LEROBOT_HOME/LIBERO-Safety/libero_safety/norm_stats.json``. If the
downloaded dataset does not include this file, place the computed statistics
there. Set the model checkpoint paths as appropriate before training.

Training
--------

OpenPI SFT and PPO:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_safety_sft_openpi_pi05
   bash examples/embodiment/run_embodiment.sh libero_safety_ppo_openpi_pi05

RLT Stage 1 and Stage 2:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_safety_rlt_stage1_sft_openpi_pi05
   bash examples/embodiment/run_embodiment.sh libero_safety_rlt_stage2_ac_mlp

Point ``rollout.rlt_feature_model.model_path`` in Stage 2 at the Stage 1 actor
checkpoint. A successful task receives ``reward_coef``; each step that violates
any BDDL constraint subtracts ``safety_cost_coef``. The environment also emits
``safety_cost`` and episodic ``safety_violation_once`` metrics. Set
``task_suite_name`` to one upstream suite to train it separately.

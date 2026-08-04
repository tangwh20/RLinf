OGPO on D4RL Adroit Relocate
================================

This example ports the official low-dimensional OGPO Relocate recipe to
RLinf.  It uses the configuration
``examples/embodiment/config/d4rl_ogpo_relocate.yaml``.

Algorithm alignment
-------------------

The actor consumes the raw 39-dimensional state and predicts a flattened
four-action chunk with a four-layer 512-unit GELU MLP.  Time uses the official
32-dimensional sinusoidal embedding.  The critic is an ensemble of ten
independent four-layer 512-unit GELU MLPs with LayerNorm.

Training has two phases.  Flow-matching behavior cloning runs for at most
50,000 updates and can stop when SDE success reaches 0.45.  Online training
then runs for 500,000 primitive transitions.  It starts updating after 10,000
primitive transitions, performs four critic and one actor update per
transition, and uses the official actor-target-before-critic update order.
Replay stores primitive transitions from completed episodes and samples
arbitrary-start four-step sequences.  ``valid`` masks actions and Q losses
that cross a terminal.  Successful episodes provide BC regularization once
the official readiness threshold is reached.

RLinf executes 32 training environments through PyTorch/Ray, so wall-clock
ordering and random samples are not bitwise identical to the
single-environment JAX reference.  The network, losses, schedules, replay
semantics, optimizer settings, and effective per-transition UTD ratios match
the official Adroit recipe.

Launch
------

Install the D4RL environment first:

.. code-block:: bash

   bash requirements/install.sh embodied --env d4rl

Place the converted expert dataset at
``datasets/relocate-expert-minari-v2.hdf5`` relative to the repository parent,
or set ``OGPO_RELOCATE_DATASET`` to its absolute path. Then run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh d4rl_ogpo_relocate

To select one physical GPU (for example GPU 1), run:

.. code-block:: bash

   OGPO_GPU_PLACEMENT=1 bash examples/embodiment/run_embodiment.sh d4rl_ogpo_relocate

``OGPO_GPU_PLACEMENT`` is the Ray hardware rank.


The default configuration uses TensorBoard.  Training metrics appear under
``train/``; evaluation uses ``eval/`` for ODE and ``eval_sde/`` for SDE.
Useful OGPO diagnostics include ``ogpo/bc_update_steps``,
``ogpo/online_env_steps``, actor PPO/BC statistics, critic TD statistics, and
success-buffer size/readiness.

Checkpoints and evaluation
--------------------------

A checkpoint is written at the BC-to-online transition, every 100,000 online
primitive transitions, and at the end.  Each checkpoint contains current and
target models, both optimizer/scheduler states, replay data, partial sequence
windows, success-buffer state, and OGPO phase counters.  Checkpoints are under
``<log_path>/<experiment_name>/checkpoints/global_step_N``.

Resume by setting ``runner.resume_dir`` to that directory.  For standalone
ODE and SDE evaluation, set the same ``runner.resume_dir`` and
``runner.only_eval=true``.  Do not point ``resume_dir`` directly at its
``actor`` subdirectory.

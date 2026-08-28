# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Robomimic environment support.

The environment adapter is imported lazily by :func:`rlinf.envs.get_env_cls` so
dataset-only processes do not initialize simulator dependencies.
"""

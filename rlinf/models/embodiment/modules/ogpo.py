# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model components used exclusively by state-based OGPO."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class SinusoidalTimeEmbedding(nn.Module):
    """Official OGPO scalar-time embedding followed by a two-layer MLP."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        if embed_dim < 4 or embed_dim % 2:
            raise ValueError("embed_dim must be an even integer >= 4")
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        if times.ndim > 1 and times.shape[-1] == 1:
            times = times.squeeze(-1)
        half_dim = self.embed_dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, device=times.device, dtype=times.dtype)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        angles = times.unsqueeze(-1) * frequencies
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=-1))


class OGPOFlowMLPActor(nn.Module):
    """PyTorch counterpart of official OGPO's ``ActorVectorField`` MLP."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        denoising_steps: int = 4,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        noise_std_head: bool = False,
        noise_std_train: float = 0.3,
        noise_std_rollout: float = 0.02,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        time_embedding_dim: int = 32,
        use_tapered_noise: bool = False,
        error_correct_sde_to_ode: bool = False,
        randn_clip_value: float = 3.0,
        clip_intermediate_actions: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.noise_std_head = noise_std_head
        self.noise_std_train = noise_std_train
        self.noise_std_rollout = noise_std_rollout
        self.use_tapered_noise = use_tapered_noise
        self.error_correct_sde_to_ode = error_correct_sde_to_ode
        self.randn_clip_value = randn_clip_value
        self.clip_intermediate_actions = clip_intermediate_actions
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))
        self.official_time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        dims = [obs_dim + action_dim + time_embedding_dim, *hidden_dims, action_dim]
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.GELU())
        self.velocity_net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def predict_velocity(self, obs, actions, timesteps, train=False):
        del train
        time_features = self.official_time_embedding(timesteps)
        return self.velocity_net(torch.cat([obs, actions, time_features], dim=-1))

    def _transition(self, obs, current, times, *, train):
        noise_std = self.noise_std_train if train else self.noise_std_rollout
        velocity = self.predict_velocity(obs, current, times, train=train)
        sigma = noise_std
        if self.use_tapered_noise:
            sigma = noise_std * torch.sqrt(torch.clamp(1.0 - times, min=0.0))
        drift = velocity
        if self.error_correct_sde_to_ode:
            if self.use_tapered_noise:
                correction = 0.5 * noise_std**2 * (times * velocity - current)
            else:
                correction = (
                    0.5
                    * noise_std**2
                    * (times * velocity - current)
                    / torch.clamp(1.0 - times, min=1e-6)
                )
            drift = drift + correction
        next_mean = current + drift / self.denoising_steps
        if self.clip_intermediate_actions:
            next_mean = next_mean.clamp(-1.0, 1.0)
        return next_mean, sigma

    def sample_chain(self, obs, train=True):
        """Sample an SDE chain and return its action and path log-probability."""
        if self.noise_std_head:
            raise NotImplementedError(
                "OGPO chain sampling currently requires fixed action noise."
            )
        batch_size = obs.shape[0]
        current = torch.randn(
            (batch_size, self.action_dim), device=obs.device, dtype=obs.dtype
        )
        chain = [current]
        for step in range(self.denoising_steps):
            times = torch.full(
                (batch_size, 1),
                step / self.denoising_steps,
                device=obs.device,
                dtype=obs.dtype,
            )
            next_mean, sigma = self._transition(obs, current, times, train=train)
            noise = torch.randn_like(current).clamp(
                -self.randn_clip_value, self.randn_clip_value
            )
            current = next_mean + sigma * noise
            if step == self.denoising_steps - 1:
                current = current.clamp(-1.0, 1.0)
            chain.append(current)

        chain_tensor = torch.stack(chain, dim=1)
        action = current.clamp(-1.0, 1.0) * self.action_scale + self.action_bias
        log_prob = self.evaluate_chain_log_prob(obs, chain_tensor, train=train)
        return action, chain_tensor, log_prob

    def sample_ode(self, obs):
        """Official evaluation ODE: Euler flow without intermediate clipping."""
        current = torch.randn(
            (obs.shape[0], self.action_dim), device=obs.device, dtype=obs.dtype
        )
        for step in range(self.denoising_steps):
            times = torch.full(
                (obs.shape[0], 1),
                step / self.denoising_steps,
                device=obs.device,
                dtype=obs.dtype,
            )
            current = (
                current
                + self.predict_velocity(obs, current, times, train=False)
                / self.denoising_steps
            )
        action = current.clamp(-1.0, 1.0) * self.action_scale + self.action_bias
        log_prob = torch.zeros((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        return action, log_prob

    def evaluate_chain_log_prob(self, obs, chain, train=True):
        """Evaluate an SDE chain under this actor with parameter gradients."""
        if chain.ndim != 3 or chain.shape[1] != self.denoising_steps + 1:
            raise ValueError(
                "chain must have shape [batch, denoising_steps + 1, action_dim]"
            )
        if self.noise_std_head:
            raise NotImplementedError(
                "OGPO chain log-probability currently requires fixed action noise."
            )
        initial = chain[:, 0]
        log_prob = (
            Normal(torch.zeros_like(initial), torch.ones_like(initial))
            .log_prob(initial)
            .sum(dim=-1)
        )
        for step in range(self.denoising_steps):
            current = chain[:, step]
            times = torch.full(
                (chain.shape[0], 1),
                step / self.denoising_steps,
                device=obs.device,
                dtype=obs.dtype,
            )
            next_mean, sigma = self._transition(obs, current, times, train=train)
            log_prob = log_prob + Normal(next_mean, sigma).log_prob(
                chain[:, step + 1]
            ).sum(dim=-1)
        return log_prob

    def forward(self, obs, train=False, log_grad=False):
        del log_grad
        action, _, log_prob = self.sample_chain(obs, train=train)
        return action, log_prob.unsqueeze(-1) if log_prob.ndim == 1 else log_prob


class OGPOMultiQHead(nn.Module):
    """Vectorized GELU MLP ensemble matching official OGPO's critic."""

    _LINEAR_MODULE_INDICES = (0, 3, 6, 9, 12)
    _NORM_MODULE_INDICES = (1, 4, 7, 10)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        num_q_heads: int = 10,
        vectorized_batch_limit: int = 1024,
    ):
        super().__init__()
        if len(hidden_dims) != 4:
            raise ValueError(
                "OGPOMultiQHead expects the official four hidden layers, got "
                f"{len(hidden_dims)}"
            )
        self.num_q_heads = num_q_heads
        self.vectorized_batch_limit = vectorized_batch_limit
        dims = [obs_dim + action_dim, *hidden_dims, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.norm_weights = nn.ParameterList()
        self.norm_biases = nn.ParameterList()
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            weight = nn.Parameter(torch.empty(num_q_heads, output_dim, input_dim))
            bias = nn.Parameter(torch.zeros(num_q_heads, output_dim))
            for head_weight in weight:
                nn.init.xavier_uniform_(head_weight)
            self.weights.append(weight)
            self.biases.append(bias)
            if index < len(dims) - 2:
                self.norm_weights.append(
                    nn.Parameter(torch.ones(num_q_heads, output_dim))
                )
                self.norm_biases.append(
                    nn.Parameter(torch.zeros(num_q_heads, output_dim))
                )

    def _forward_vectorized(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs.unsqueeze(0).expand(self.num_q_heads, -1, -1)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = torch.bmm(activations, weight.transpose(1, 2))
            activations = activations + bias[:, None, :]
            if index < len(self.norm_weights):
                normalized_shape = (activations.shape[-1],)
                activations = torch.vmap(
                    lambda value, norm_weight, norm_bias: F.layer_norm(
                        value, normalized_shape, norm_weight, norm_bias, 1e-5
                    )
                )(
                    activations,
                    self.norm_weights[index],
                    self.norm_biases[index],
                )
                activations = F.gelu(activations)
        return activations.squeeze(-1).transpose(0, 1)

    def _forward_head(self, inputs: torch.Tensor, head_index: int) -> torch.Tensor:
        activations = inputs
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = F.linear(activations, weight[head_index], bias[head_index])
            if index < len(self.norm_weights):
                activations = F.layer_norm(
                    activations,
                    (activations.shape[-1],),
                    self.norm_weights[index][head_index],
                    self.norm_biases[index][head_index],
                    1e-5,
                )
                activations = F.gelu(activations)
        return activations

    def _forward_sequential(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self._forward_head(inputs, head_index)
                for head_index in range(self.num_q_heads)
            ],
            dim=-1,
        )

    def forward(
        self, state_features: torch.Tensor, action_features: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([state_features, action_features], dim=-1)
        if inputs.shape[0] <= self.vectorized_batch_limit:
            return self._forward_vectorized(inputs)
        return self._forward_sequential(inputs)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load both vectorized and legacy ``qs.<head>`` checkpoints."""
        first_legacy_key = f"{prefix}qs.0.0.weight"
        if first_legacy_key in state_dict and f"{prefix}weights.0" not in state_dict:
            for layer_index, module_index in enumerate(self._LINEAR_MODULE_INDICES):
                for parameter_name, destination in (
                    ("weight", "weights"),
                    ("bias", "biases"),
                ):
                    legacy_keys = [
                        f"{prefix}qs.{head_index}.{module_index}.{parameter_name}"
                        for head_index in range(self.num_q_heads)
                    ]
                    state_dict[f"{prefix}{destination}.{layer_index}"] = torch.stack(
                        [state_dict.pop(key) for key in legacy_keys]
                    )
            for layer_index, module_index in enumerate(self._NORM_MODULE_INDICES):
                for parameter_name, destination in (
                    ("weight", "norm_weights"),
                    ("bias", "norm_biases"),
                ):
                    legacy_keys = [
                        f"{prefix}qs.{head_index}.{module_index}.{parameter_name}"
                        for head_index in range(self.num_q_heads)
                    ]
                    state_dict[f"{prefix}{destination}.{layer_index}"] = torch.stack(
                        [state_dict.pop(key) for key in legacy_keys]
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional
from torchrl.modules import MaskedCategorical
from .mlp import MLP
from .gin_backup import GraphCNN
import numpy as np



def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            n_j: int,
            n_m: int,
            num_layers: int,
            hidden_dim: int,
            learn_eps: bool,
            input_dim: int,
            num_mlp_layers_feature_extract: int,
            neighbor_pooling_type: str,
            device: str,
            dropout: float = 0.0,
                 ):
        super().__init__()
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device,
                                        dropout=dropout).to(device)

    def forward(self, adj, features, candidate, graph_pool):
        h_pooled, h_nodes = self.feature_extract(
            x=features,
            graph_pool=graph_pool,
            padded_nei=None,
            adj=adj)

        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)

        return concateFea, h_pooled


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor) -> MaskedCategorical:
        dist = MaskedCategorical(logits=self.net(obs), mask=action_mask)
        return dist

    @torch.no_grad()
    def act(self, state: np.ndarray, action_mask: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # if action_mask is not None:
        masks = torch.as_tensor(action_mask.reshape(1, -1), device=device, dtype=torch.bool)
        dist = self(state, masks)
        if self.training:
            action = dist.sample()
            # print("hij sampled")
        else:
            action = torch.argmax(dist.probs)
        # print(action)
        return action


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.4
    ):
        super().__init__()
        self.action_dim = action_dim
        dims = [state_dim, *([hidden_dim] * n_hidden), action_dim]
        self.q1 = MLP(dims, dropout=dropout)
        self.q2 = MLP(dims, dropout=dropout)

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output1 = self.q1(state)
        output2 = self.q2(state)
        q1 = torch.gather(output1, 1, action.long()).flatten()
        q2 = torch.gather(output2, 1, action.long()).flatten()
        # sa = torch.cat([state, action], 1)
        return q1, q2

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class CategoricalPolicyL2D(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
        activation_fn: Callable[[], nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), 1],
            dropout=dropout,
            activation_fn=activation_fn,
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor) -> MaskedCategorical:
        out_net = self.net(obs).squeeze(-1)

        dist = MaskedCategorical(logits=out_net, mask=action_mask)
        return dist

    @torch.no_grad()
    def act(self, state: torch.Tensor, action_mask: np.ndarray, device: str = "cpu", deterministic: bool = False):
        # state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # if action_mask is not None:
        masks = torch.as_tensor(action_mask.reshape(1, -1), device=device, dtype=torch.bool)
        dist = self(state, masks)
        if deterministic:
            action = torch.argmax(dist.probs)

            # print("hij sampled")
        else:
            action = dist.sample()

        # print(action)
        return action


class TwinQL2D(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2,
            activation_fn: Callable[[], nn.Module] = nn.Tanh, dropout: float = 0.4
    ):
        super().__init__()
        self.action_dim = action_dim
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, activation_fn=activation_fn, dropout=dropout)
        self.q2 = MLP(dims, activation_fn=activation_fn, dropout=dropout)

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output1 = self.q1(state).squeeze(-1)
        output2 = self.q2(state).squeeze(-1)
        # print("output1", output1.shape)
        # print("action", action.shape)
        # exit()
        q1 = torch.gather(output1, 1, action.long()).flatten()
        q2 = torch.gather(output2, 1, action.long()).flatten()
        # sa = torch.cat([state, action], 1)
        return q1, q2

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunctionL2D(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2,
                 activation_fn: Callable[[], nn.Module] = nn.Tanh, dropout: float = 0.0):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True, activation_fn=activation_fn, dropout=dropout)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class CategoricalPolicySACDiscrete(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden_layers), act_dim],
        )
        if orthogonal_init:
            self.net.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.net[-1], False)

    def log_prob(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        net_out = self.net(obs)
        dist = MaskedCategorical(logits=net_out, mask=action_mask)
        action_probs = dist.probs
        z = action_probs == 0.0
        z = z.float() * 1e-8
        action_log_probs = torch.log(action_probs + z)
        return action_log_probs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def evaluate(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print("obs", obs.shape)
        net_out = self.net(obs)
        if action_mask is None:
            action_mask = torch.ones_like(net_out).bool()
        dist = MaskedCategorical(logits=net_out, mask=action_mask)
        action = dist.sample()
        action_probs = dist.probs
        z = action_probs == 0.0
        z = z.float() * 1e-8
        action_log_probs = torch.log(action_probs + z)
        return action, action_probs, action_log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, action_mask: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        masks = torch.as_tensor(action_mask.reshape(1, -1), device=device, dtype=torch.bool)
        dist = MaskedCategorical(logits=self.forward(state), mask=masks)
        if self.training:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs)
        return action


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.observation_dim = observation_dim

        self.orthogonal_init = orthogonal_init

        dims = [observation_dim, *([hidden_dim] * n_hidden_layers), action_dim]

        self.network = MLP(dims)

        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        q_values = torch.squeeze(self.network(observations), dim=-1)
        return q_values

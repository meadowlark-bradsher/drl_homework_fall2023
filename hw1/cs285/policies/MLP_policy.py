"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp

class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def policy(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes the mean action vector for a given observation using the mean network.

        :param observation: observation(s) to query the policy
        :return:
            mean_action: predicted mean action vector (continuous values)
        """
        mean_action = self.mean_net(observation)
        return mean_action

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    # def forward(self, observation: torch.FloatTensor) -> torch.FloatTensor:
    #     """
    #     Defines the forward pass of the network for continuous actions.
    #
    #     :param observation: observation(s) to query the policy
    #     :return:
    #         mean_action: predicted continuous action vector
    #     """
    #     # Pass observation through the mean network to get the mean action
    #     mean_action = self.mean_net(observation)
    #     return mean_action

    def forward(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        """
        Defines the forward pass of the network for continuous actions.

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy (continuous values)
        """
        # Pass observation through the mean network to get the mean action
        mean_action = self.mean_net(observation)

        # Optional: Sample from a Gaussian distribution if you want to introduce stochasticity
        # (Here, we're assuming self.logstd has been defined to represent log standard deviation)
        std = torch.exp(self.logstd)
        distribution = torch.distributions.Normal(mean_action, std)
        action = distribution.rsample()  # Use rsample() for reparameterization trick (if training with gradients)

        return action  # This is the continuous action to be used

    def update(self, observations, actions):
        """
        Updates/trains the policy by comparing predicted actions to expert actions.

        :param observations: observation(s) to query the policy (numpy array)
        :param actions: actions we want the policy to imitate (numpy array)
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # Convert observations and actions to torch tensors
        observations = torch.from_numpy(observations).float().to(ptu.device)  # Convert to tensor and move to device
        actions = torch.from_numpy(actions).float().to(ptu.device)  # Convert to tensor and move to device

        # Forward pass to predict actions given observations
        predicted_actions = self.forward(observations)

        # Compute loss (e.g., Mean Squared Error for continuous actions)
        loss = torch.nn.functional.mse_loss(predicted_actions, actions)

        # Backpropagation
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update parameters

        return {
            'Training Loss': loss.item(),
        }

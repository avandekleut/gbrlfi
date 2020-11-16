import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import copy
torch.set_default_dtype(torch.float32)
class ConcatenationEncoder(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        observation_dim = observation_space['observation'].shape[0]
        desired_goal_dim = observation_space['desired_goal'].shape[0]
        self.output_dim = observation_dim + desired_goal_dim
        
    def forward(self, observation, desired_goal, detach=False):
        observation = torch.cat([observation, desired_goal], -1)

        return observation

class ConvolutionalEncoder(nn.Module):
    def __init__(self, observation_space, feature_dim):
        """
        Assumes that the observation and goal images have the same dimensions.
        """
        super().__init__()
        
        assert observation_space['image_observation'].shape == (3, 84, 84)
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # compute output of flattened conv
        with torch.no_grad():
            n_flatten = self.conv(
                torch.as_tensor(observation_space['image_observation'].sample()).float().unsqueeze(0)
            ).shape[-1]
        
        self.head = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.LayerNorm(feature_dim))
        
        self.output_logits = False
        self.output_dim = 2 * feature_dim # double since outputs get concatenated
        
    def forward_single_observation(self, observation, detach=False):
        observation = observation / 255.
        
        observation = self.conv(observation)
        
        if detach:
            observation = observation.detach()
            
        observation = self.head(observation)
        
        return observation
        
    def forward(self, observation, desired_goal, detach=False):
        observation = self.forward_single_observation(observation, detach=detach)
        desired_goal = self.forward_single_observation(desired_goal, detach=detach)
        
        observation = torch.cat([observation, desired_goal], -1)
        
        return observation
    
    def copy_conv_weights_from(self, source):
        self.conv.load_state_dict(source.conv.state_dict())
        
class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
            self,
            from_images,
            observation_space,
            action_space,
            feature_dim, # ignored for state-based
            hidden_sizes,
            log_std_bounds):
        super().__init__()

        if from_images:
            self.encoder = ConvolutionalEncoder(observation_space, feature_dim)
        else:
            self.encoder = ConcatenationEncoder(observation_space)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.output_dim, 2 * action_space.shape[0], hidden_sizes) # twice the dimensions for outputs to split into mu and log_sigma

#         self.apply(utils.weight_init)

    def forward(self, observation, desired_goal, detach_encoder=False):
        
        observation = self.encoder(observation, desired_goal, detach=detach_encoder)
        
        mu, log_std = self.trunk(observation).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)

        return dist
        
class Critic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(
            self,
            from_images,
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes):
        super().__init__()
        
        if from_images:
            # images
            self.encoder = ConvolutionalEncoder(observation_space, feature_dim)
        else:
            # states
            self.encoder = ConcatenationEncoder(observation_space)

        self.Q1 = utils.mlp(self.encoder.output_dim + action_space.shape[0],
                            1, hidden_sizes)
        self.Q2 = utils.mlp(self.encoder.output_dim + action_space.shape[0],
                            1, hidden_sizes)

#         self.apply(utils.weight_init)

    def forward(self, observation, desired_goal, action, detach_encoder=False):
        assert observation.size(0) == action.size(0) # batch sizes match. critic isn't used on non-batches.
        observation = self.encoder(observation, desired_goal, detach=detach_encoder)

        observation = torch.cat([observation, action], dim=-1)
        q1 = self.Q1(observation)
        q2 = self.Q2(observation)

        return q1, q2
    
class Agent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""

    def __init__(self,
            from_images,
            observation_space,
            action_space,
            feature_dim=32,
            hidden_sizes=[256, 256],
            log_std_bounds=[-10, 2],
            discount=0.95,
            init_temperature=0.1,
            lr=1e-3,
            actor_update_frequency=2,
            critic_tau=0.005,
            critic_target_update_frequency=2,
            batch_size=256,
            device='cuda',
        ):
        self.action_range = (
            float(action_space.low.min()),
            float(action_space.high.max())
        )
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = Actor(
            from_images,
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes,
            log_std_bounds).to(
            self.device)

        self.critic = Critic(
            from_images,
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes).to(
            self.device)
        
        self.critic_target = copy.deepcopy(self.critic)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad = False

        # tie conv layers between actor and critic
        if from_images:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_space.shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.info = dict()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, observation, desired_goal, sample=False):
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        desired_goal = torch.FloatTensor(desired_goal).to(self.device).unsqueeze(0)
        dist = self.actor(observation, desired_goal)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, observation, desired_goal, action, reward, next_observation, not_done, step):
        # batch
        
        with torch.no_grad():
            dist = self.actor(next_observation, desired_goal)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True) # log prob for diagonal gaussian
            target_Q1, target_Q2 = self.critic_target(next_observation, desired_goal, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)
            
            target_Q = torch.clamp(target_Q, -float('inf'), 0)
#             target_Q = torch.clamp(target_Q, -2./(1. - self.discount), 0) # from HER paper, modified for our reward function
        
        current_Q1, current_Q2 = self.critic(observation, desired_goal, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.info['critic_loss'] = critic_loss.item()

    def update_actor_and_alpha(self, observation, desired_goal, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(observation, desired_goal, detach_encoder=True)
        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True) # diagonal gaussian

        # We dont need grad wrt critic theta since we aren't optimizing them anyways.
        # This saves some computational effort.
        for parameter in self.critic.parameters():
            parameter.requires_grad = False
        
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(observation, desired_goal, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        for parameter in self.critic.parameters():
            parameter.requires_grad = True
        
        self.info['actor_loss'] = actor_loss.item()
        self.info['entropy'] = -log_prob.mean().item()
        self.info['alpha_loss'] = alpha_loss.item()
        self.info['alpha'] = self.alpha.item()
        self.info['actor_Q'] = actor_Q.mean().item()

    def update(self, replay_buffer, step):
        observation, desired_goal, action, reward, next_observation, not_done = replay_buffer.sample(self.batch_size)
        
        self.update_critic(observation, desired_goal, action, reward, next_observation, not_done, step)
        
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(observation, desired_goal, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
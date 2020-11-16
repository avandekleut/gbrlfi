import torch
import torch.nn as nn
from torch import distributions as pyd
import torch.nn.functional as F
import numpy as np
import math
import random

def get_goal_image(env, goal, width=84, height=84, solve_iters=100):
    """
    Given a goal, set the simulator state to the goal.
    This is used to render snapshots of the goal.
    Goals are always xyz coordinates, either of gripper end effector or of object.
    """
    if env.has_object:
        object_qpos = env.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = goal
        env.sim.data.set_joint_qpos('object0:joint', object_qpos)
    env.sim.data.set_mocap_pos('robot0:mocap', goal)
    env.sim.forward()
    for solve_iter in range(solve_iters):
        env.sim.step()
    obs_img = env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
    assert obs_img.shape == env.observation_space['image_observation'].shape
    return obs_img

def mlp(input_dimensions, output_dimensions, hidden_sizes=[400, 300], activation=nn.ReLU(), out_activation=None):
    layer_sizes = [input_dimensions] + hidden_sizes + [output_dimensions]
    modules = []
    for i in range(len(layer_sizes)-1):
        modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        modules.append(activation)
    modules.pop() # remove last activation
    if out_activation is not None:
        modules.append(out_activation)
    return nn.Sequential(*modules)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
    
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
            
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
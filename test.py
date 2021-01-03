# Before running, install required packages:
# pip install tianshou
import os
import gym
import torch
import pprint
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

if __name__ == '__main__':
    # in this folder will be saved the best model and/or tensorboard files
    logdir = "log"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = "Acrobot-v1"
    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.env.action_space.shape or env.env.action_space.n
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: gym.make(task)
                                   for _ in range(16)])
    test_envs = SubprocVectorEnv([lambda: gym.make(task)
                                  for _ in range(10)])
    # seed
    np.random.seed(0)
    torch.manual_seed(0)
    train_envs.seed(0)
    test_envs.seed(0)
    # define model
    layers_num = 3
    net = Net(layers_num, state_shape,
              action_shape, device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    # define policy
    policy = DQNPolicy(net, optim, discount_factor=0.99,
                       estimation_step=3,
                       target_update_freq=300)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = ReplayBuffer(20000)    # collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(logdir, 'CartPole-v1', 'DQN')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy, os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):
        eps_train = 0.1
        eps_train_final = 0.05
        linear_decay_steps = 50000
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= linear_decay_steps:
            eps = eps_train - env_step / linear_decay_steps * \
                (eps_train - eps_train_final)
        else:
            eps = eps_train_final
        policy.set_eps(eps)
        writer.add_scalar('train/eps', eps, global_step=env_step)

    def test_fn(epoch, env_step):
        policy.set_eps(0.005)

    # watch agent's performance
    def watch():
        print("Testing agent ...")
        policy.eval()
        policy.set_eps(0.005)
        test_envs.seed(0)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * 10,
                                        render=0)
        pprint.pprint(result)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=64 * 4)
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10,
        step_per_epoch=1000,
        collect_per_step=100,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        save_fn=save_fn,
        writer=writer,
        test_in_train=False)

    pprint.pprint(result)
    watch()
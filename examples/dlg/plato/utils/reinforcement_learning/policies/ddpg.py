"""
Reference:

https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
"""
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.utils.reinforcement_learning.policies import base


class Policy(base.Policy):
    def __init__(self, state_dim, action_space):
        super().__init__(state_dim, action_space)

    def select_action(self, state):
        """ Select action from policy. """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        """ Update policy. """
        for _ in range(Config().algorithm.update_iteration):
            # Sample replay buffer
            state, action, reward, next_state, done = self.replay_buffer.sample(
            )
            state = torch.FloatTensor(state).to(self.device).unsqueeze(1)
            action = torch.FloatTensor(action).to(self.device).unsqueeze(1)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            next_state = torch.FloatTensor(next_state).to(
                self.device).unsqueeze(1)
            done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

            # Compute the target Q value
            target_Q = self.critic_target(next_state,
                                          self.actor_target(next_state))
            target_Q = reward + (
                (1 - done) * Config().algorithm.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(Config().algorithm.tau * param.data +
                                        (1 - Config().algorithm.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(Config().algorithm.tau * param.data +
                                        (1 - Config().algorithm.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

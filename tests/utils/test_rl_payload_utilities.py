"""Active RL policy training, batch isolation, and persistence regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from plato.config import Config
from plato.utils.reinforcement_learning.policies import ddpg, sac, td3


@pytest.fixture
def action_space(request):
    """Keep the numeric contract in base and qualify real Box in mandatory."""
    if request.config.getoption("test_profile") == "mandatory":
        from gymnasium import spaces

        return spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    return SimpleNamespace(
        shape=(2,),
        low=np.full(2, -1, dtype=np.float32),
        high=np.full(2, 1, dtype=np.float32),
    )


@pytest.fixture
def rl_config(temp_config, monkeypatch):
    config = SimpleNamespace(
        max_action=1,
        learning_rate=0.001,
        replay_size=16,
        replay_seed=17,
        batch_size=3,
        update_iteration=2,
        gamma=0.99,
        tau=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,
        recurrent_actor=False,
        hidden_size=8,
        deterministic=False,
        automatic_entropy_tuning=True,
        alpha=0.2,
        epsilon=1e-6,
        model_name="td3",
    )
    monkeypatch.setattr(Config, "algorithm", config)
    return config


@pytest.mark.parametrize("kind", ["ddpg", "td3", "sac", "td3_rnn", "sac_deterministic"])
def test_active_policy_finite_update_and_saved_weight_roundtrip(
    rl_config, action_space, kind
):
    torch.manual_seed(17)
    np.random.seed(17)
    if kind == "sac_deterministic":
        rl_config.deterministic = True
    if kind == "td3_rnn":
        rl_config.recurrent_actor = True
    if kind.startswith("sac"):
        policy = sac.Policy(4, action_space)
    else:
        policy = (ddpg if kind == "ddpg" else td3).Policy(4, 2)
    for idx in range(6):
        state = np.array([idx / 10, 0.1, -0.2, 0.4])
        record = (
            state,
            np.array([0.25, 0.75]),
            idx / 10,
            state + 0.05,
            float(idx == 5),
        )
        if kind == "td3_rnn":
            h, c = policy.get_initial_states()
            record += (h, c, h, c)
        policy.replay_buffer.push(record)
    before = {name: p.detach().clone() for name, p in policy.actor.named_parameters()}
    losses = policy.update()
    assert all(np.isfinite(loss) for loss in losses)
    assert any(
        not torch.equal(p, before[name]) for name, p in policy.actor.named_parameters()
    )
    assert all(torch.isfinite(p).all() for p in policy.actor.parameters())
    if kind == "sac_deterministic":
        assert policy.alpha == 0
    saved = {name: p.detach().clone() for name, p in policy.actor.named_parameters()}
    policy.save_model(1)
    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(1)
    policy.load_model(1)
    for name, p in policy.actor.named_parameters():
        torch.testing.assert_close(p, saved[name])
    # Target networks must follow a loaded snapshot too.
    for p, target in zip(policy.critic.parameters(), policy.critic_target.parameters()):
        torch.testing.assert_close(p, target)


def test_td3_actions_are_independent_of_other_batch_rows(rl_config):
    torch.manual_seed(17)
    actor = td3.TD3Actor(4, 2, 1)
    states = torch.tensor([[0.1, -0.2, 0.3, 0.4], [1.0, 2.0, -1.0, 0.2]])
    batched = actor(states)
    isolated = torch.cat([actor(state[None, :]) for state in states])
    torch.testing.assert_close(batched, isolated)
    torch.testing.assert_close(batched.sum(-1), torch.ones(2))


@pytest.mark.parametrize("batch_size", [1, 3])
def test_async_recurrent_policy_accepts_variable_client_sequences(
    rl_config, monkeypatch, batch_size
):
    monkeypatch.setattr(Config, "server", SimpleNamespace(synchronous=False))
    rl_config.recurrent_actor = True
    rl_config.batch_size = batch_size
    policy = td3.Policy(4, 3)
    h, c = policy.get_initial_states()
    for length in [1, 2, 3]:
        state = np.arange(length * 4).reshape(length, 4) / 10
        policy.replay_buffer.push(
            (state, np.full((length, 1), 1 / length), 0.1, state + 0.01, 0, h, c, h, c)
        )
    losses = policy.update()
    assert all(np.isfinite(loss) for loss in losses)


def test_td3_target_actions_use_every_transition(rl_config, monkeypatch):
    torch.manual_seed(17)
    rl_config.policy_noise = 0
    policy = td3.Policy(4, 2)
    state = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, -1, 0.5, 0.6], [0.7, 0.8, 0.9, 1]])
    action = np.full((3, 2), 0.5)
    reward, done = np.arange(3).reshape(3, 1) / 10, np.zeros((3, 1))
    monkeypatch.setattr(
        policy.replay_buffer,
        "sample",
        lambda: (state, action, reward, state + 0.2, done),
    )
    expected = policy.actor_target(torch.tensor(state + 0.2, dtype=torch.float32))
    seen = []

    def observe(_module, inputs):
        seen.append(inputs[1].detach().clone())

    handle = policy.critic_target.register_forward_pre_hook(observe)
    try:
        policy.update()
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], expected)


def test_async_actor_ignores_padding_and_other_transitions(rl_config, monkeypatch):
    monkeypatch.setattr(Config, "server", SimpleNamespace(synchronous=False))
    torch.manual_seed(17)
    actor = td3.RNNActor(4, 3, 8, 1)
    states = [torch.ones(1, 4), torch.arange(8).reshape(2, 4).float()]
    before = [state.clone() for state in states]
    batched, _hidden = actor(states)
    isolated = torch.cat([actor([state])[0] for state in states])
    torch.testing.assert_close(batched, isolated)
    torch.testing.assert_close(batched.sum((1, 2)), torch.ones(2))
    assert torch.count_nonzero(batched[0, 1:]) == 0
    assert torch.count_nonzero(batched[1, 2:]) == 0
    for actual, expected in zip(states, before):
        torch.testing.assert_close(actual, expected)

from __future__ import annotations

from AutoAvoider.sim.mock_env import MockSimEnvironment


def test_mock_env_reset_step() -> None:
    env = MockSimEnvironment(width=64, height=32, max_steps=3)
    obs = env.reset()
    assert "stereo/left" in obs
    assert "stereo/right" in obs

    done = False
    steps = 0
    while not done:
        obs, info, done, meta = env.step({"throttle": 0.1, "steer": -0.1})
        steps += 1
        assert "speed" in info
        assert "steer" in info
        assert "step" in meta

    assert steps == 3

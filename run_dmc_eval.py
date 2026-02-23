"""
Script to evaluate a trained Dreamer/R2-Dreamer checkpoint on DMC.
Example usage:
python run_dmc_eval.py   --checkpoint logdir/test/latest.pt   --episodes 10   --device cuda:0   --override env=dmc_proprio   --override env.task=dmc_reacher_easy --video logdir/test/eval_reacher.mp4 --fps 30

"""


import argparse
import pathlib

import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

import tools
from dreamer import Dreamer
from envs import make_env
import matplotlib.pyplot as plt


def _obs_to_tensordict(obs, device):
    tensors = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, device=device)
        # Add a batch dimension for a single environment.
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        else:
            tensor = tensor.unsqueeze(0)
        tensors[key] = tensor
    return TensorDict(tensors, batch_size=(1,), device=device)


def _load_config(checkpoint_path):
    config_path = checkpoint_path.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find run config: {config_path}")
    return OmegaConf.load(config_path)


def _compose_config(overrides):
    config_dir = pathlib.Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="configs", overrides=overrides)


def evaluate(args):
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.override:
        config = _compose_config(args.override)
    else:
        config = _load_config(checkpoint_path)
    if args.task is not None:
        config.env.task = args.task
    if args.device is not None:
        config.device = args.device
        config.model.device = args.device
        config.env.device = args.device
    # We do not train in this script, so compiling the update path is unnecessary.
    config.model.compile = False

    device = torch.device(config.device)
    tools.set_seed_everywhere(int(config.seed))
    torch.set_float32_matmul_precision("high")

    env = make_env(config.env, id=0)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    if "agent_state_dict" not in payload:
        raise KeyError("Checkpoint does not contain 'agent_state_dict'.")
    try:
        agent.load_state_dict(payload["agent_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint with the current config. "
            "If this logdir was reused, pass the original Hydra overrides, e.g. "
            "--override env=dmc_proprio --override env.task=dmc_reacher_easy."
        ) from exc
    # Make sure frozen inference modules are synced.
    agent.clone_and_freeze()
    agent.eval()

    episode_returns = []
    episode_lengths = []
    all_frames = []

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        state = agent.get_initial_state(1)

        if args.video is not None and "image" in obs:
            all_frames.append(obs["image"])

        while not done and ep_len < args.max_steps:
            trans = _obs_to_tensordict(obs, device)
            with torch.no_grad():
                action, state = agent.act(trans, state, eval=True)
            action_np = tools.to_np(action[0]).astype(np.float32)
            obs, reward, done, _ = env.step(action_np)
            ep_return += float(reward)
            ep_len += 1
            if args.video is not None and "image" in obs:
                all_frames.append(obs["image"])

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        print(f"[Episode {episode + 1:03d}] return={ep_return:.3f} length={ep_len}")

    ret_mean = float(np.mean(episode_returns))
    ret_std = float(np.std(episode_returns))
    len_mean = float(np.mean(episode_lengths))
    print(f"Mean return over {args.episodes} episodes: {ret_mean:.3f} +/- {ret_std:.3f}")
    print(f"Mean episode length: {len_mean:.1f}")

    if args.video is not None:
        video_path = pathlib.Path(args.video).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, all_frames, fps=args.fps)
        print(f"Saved evaluation video to: {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Dreamer/R2-Dreamer checkpoint on DMC.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (e.g., logdir/test/latest.pt).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Max environment steps per episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from run config (e.g., cpu or cuda:0).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional env.task override (e.g., dmc_reacher_easy).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override(s) used to rebuild config, e.g. --override env=dmc_proprio.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional output path for rendered evaluation video (.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for saved video.",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

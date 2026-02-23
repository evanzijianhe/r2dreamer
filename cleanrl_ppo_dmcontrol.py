import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import tools
from envs.dmc import DeepMindControl
from envs.wrappers import Dtype, NormalizeActions, TimeLimit


def parse_args():
    parser = argparse.ArgumentParser(description="PPO baseline for DeepMind Control Suite tasks.")
    parser.add_argument("--task", type=str, default="walker_walk", help="DMC task, e.g. walker_walk or dmc_walker_walk.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=510_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-obs", action="store_true", default=True)
    parser.add_argument("--no-norm-obs", dest="norm_obs", action="store_false")
    parser.add_argument("--norm-reward", action="store_true", default=True)
    parser.add_argument("--no-norm-reward", dest="norm_reward", action="store_false")
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=1000)
    parser.add_argument(
        "--count-env-steps",
        action="store_true",
        default=True,
        help="Count timesteps in environment steps (multiplied by action_repeat).",
    )
    parser.add_argument(
        "--count-agent-steps",
        dest="count_env_steps",
        action="store_false",
        help="Count timesteps in policy decisions (CleanRL-style).",
    )
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    parser.add_argument("--torch-deterministic", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default="logdir/ppo_dmc")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=0,
        help="Save checkpoint every N PPO updates. 0 disables periodic saves.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-deterministic", action="store_true", default=True)
    parser.add_argument("--no-eval-deterministic", dest="eval_deterministic", action="store_false")
    return parser.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = np.expand_dims(x, axis=0)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def state_dict(self):
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class VecNormalize:
    def __init__(self, obs_shape, num_envs, gamma, normalize_obs, normalize_reward, clip_obs, clip_reward, eps=1e-8):
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.eps = eps
        self.gamma = gamma
        self.returns = np.zeros(num_envs, dtype=np.float64)

        self.obs_rms = RunningMeanStd(obs_shape)
        self.ret_rms = RunningMeanStd((1,))

    def normalize_observation(self, obs, update=True):
        obs = np.asarray(obs, dtype=np.float64)
        if not self.normalize_obs:
            return obs.astype(np.float32)
        if update:
            self.obs_rms.update(obs)
        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs.astype(np.float32)

    def normalize_rewards(self, rewards, dones, update=True):
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.bool_)
        if not self.normalize_reward:
            self.returns[dones] = 0.0
            return rewards.astype(np.float32)
        self.returns = self.returns * self.gamma + rewards
        if update:
            self.ret_rms.update(self.returns.reshape(-1, 1))
        rewards = rewards / np.sqrt(self.ret_rms.var + self.eps)
        rewards = np.clip(rewards, -self.clip_reward, self.clip_reward)
        self.returns[dones] = 0.0
        return rewards.astype(np.float32)

    def state_dict(self):
        return {
            "normalize_obs": self.normalize_obs,
            "normalize_reward": self.normalize_reward,
            "clip_obs": float(self.clip_obs),
            "clip_reward": float(self.clip_reward),
            "eps": float(self.eps),
            "gamma": float(self.gamma),
            "returns": self.returns.copy(),
            "obs_rms": self.obs_rms.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
        }

    def load_state_dict(self, state):
        self.normalize_obs = bool(state["normalize_obs"])
        self.normalize_reward = bool(state["normalize_reward"])
        self.clip_obs = float(state["clip_obs"])
        self.clip_reward = float(state["clip_reward"])
        self.eps = float(state["eps"])
        self.gamma = float(state["gamma"])
        self.returns = np.asarray(state["returns"], dtype=np.float64)
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.ret_rms.load_state_dict(state["ret_rms"])


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class DMControlVectorEnv:
    def __init__(self, task, num_envs, action_repeat, time_limit, seed):
        self.num_envs = num_envs
        self.envs = [self._make_env(task, action_repeat, time_limit, seed + i) for i in range(num_envs)]
        self.single_action_space = self.envs[0].action_space

        sample_obs = self.envs[0].reset()
        excluded = {"image", "is_first", "is_last", "is_terminal"}
        self.obs_keys = sorted([k for k in sample_obs.keys() if k not in excluded])
        if not self.obs_keys:
            raise ValueError(
                "No proprioceptive keys found in observation dict. "
                "This PPO baseline expects state observations (e.g., walker_walk), not pure image observations."
            )
        self.obs_dim = sum(np.asarray(sample_obs[k], dtype=np.float32).size for k in self.obs_keys)

        self._ep_returns = np.zeros(num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)

    @staticmethod
    def _make_env(task, action_repeat, time_limit, seed):
        env = DeepMindControl(name=task, action_repeat=action_repeat, seed=seed)
        env = NormalizeActions(env)
        env = TimeLimit(env, time_limit // action_repeat)
        env = Dtype(env)
        return env

    def _flatten_obs(self, obs):
        parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in self.obs_keys]
        return np.concatenate(parts, axis=0)

    def reset(self):
        batch = []
        self._ep_returns.fill(0.0)
        self._ep_lengths.fill(0)
        for env in self.envs:
            obs = env.reset()
            batch.append(self._flatten_obs(obs))
        return np.stack(batch, axis=0)

    def step(self, actions):
        next_obs = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        episode_returns = np.full(self.num_envs, np.nan, dtype=np.float32)
        episode_lengths = np.full(self.num_envs, np.nan, dtype=np.float32)

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, _ = env.step(action)
            self._ep_returns[i] += reward
            self._ep_lengths[i] += 1

            if done:
                dones[i] = True
                episode_returns[i] = self._ep_returns[i]
                episode_lengths[i] = float(self._ep_lengths[i])
                obs = env.reset()
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0

            rewards[i] = reward
            next_obs.append(self._flatten_obs(obs))

        infos = {"episode_return": episode_returns, "episode_length": episode_lengths}
        return np.stack(next_obs, axis=0), rewards, dones, infos

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


def normalize_task(task):
    normalized = task.strip()
    if normalized.startswith("dmc_"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "_")
    if "_" not in normalized:
        raise ValueError(
            f"Invalid --task '{task}'. Expected a DeepMind Control task in domain_task format, "
            "for example: walker_walk, reacher_easy, cheetah_run."
        )
    return normalized

def save_checkpoint(path, agent, optimizer, vecnorm, args, global_step, update):
    payload = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vecnorm_state_dict": vecnorm.state_dict(),
        "args": vars(args),
        "global_step": int(global_step),
        "update": int(update),
    }
    torch.save(payload, path)


def evaluate(
    agent,
    task,
    action_repeat,
    time_limit,
    seed,
    eval_episodes,
    device,
    vecnorm,
    deterministic=True,
):
    eval_env = DMControlVectorEnv(
        task=task,
        num_envs=1,
        action_repeat=action_repeat,
        time_limit=time_limit,
        seed=seed + 10_000,
    )
    action_low = torch.as_tensor(eval_env.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(eval_env.single_action_space.high, dtype=torch.float32, device=device)

    episode_returns = []
    was_training = agent.training
    agent.eval()
    obs_np = eval_env.reset()
    obs_np = vecnorm.normalize_observation(obs_np, update=False)

    while len(episode_returns) < eval_episodes:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            if deterministic:
                action_t = agent.actor_mean(obs_t)
            else:
                action_t, _, _, _ = agent.get_action_and_value(obs_t)
            action_t = torch.clamp(action_t, action_low, action_high)
        obs_np, _, _, infos = eval_env.step(action_t.cpu().numpy())
        obs_np = vecnorm.normalize_observation(obs_np, update=False)
        ret = infos["episode_return"][0]
        if not np.isnan(ret):
            episode_returns.append(float(ret))

    if was_training:
        agent.train()
    eval_env.close()
    return episode_returns

def main():
    args = parse_args()
    task = normalize_task(args.task)

    step_scale = args.action_repeat if args.count_env_steps else 1
    ep_length_scale = args.action_repeat if args.count_env_steps else 1
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by num_minibatches={args.num_minibatches}.")

    rollout_steps = args.num_envs * args.num_steps * step_scale
    if args.total_timesteps < rollout_steps:
        raise ValueError(
            f"total_timesteps={args.total_timesteps} is smaller than one PPO rollout ({rollout_steps})."
        )

    num_updates = args.total_timesteps // rollout_steps
    minibatch_size = batch_size // args.num_minibatches

    run_name = f"ppo_dmc__{task}__seed{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    writer.add_text(
        "run/config",
        "\n".join([f"{k}: {v}" for k, v in sorted(vars(args).items())]),
    )

    tools.set_seed_everywhere(args.seed)
    if args.torch_deterministic:
        tools.enable_deterministic_run()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = DMControlVectorEnv(
        task=task,
        num_envs=args.num_envs,
        action_repeat=args.action_repeat,
        time_limit=args.time_limit,
        seed=args.seed,
    )

    obs_dim = envs.obs_dim
    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device)
    vecnorm = VecNormalize(
        obs_shape=(obs_dim,),
        num_envs=args.num_envs,
        gamma=args.gamma,
        normalize_obs=args.norm_obs,
        normalize_reward=args.norm_reward,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
    )

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim, hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs_np = envs.reset()
    next_obs_np = vecnorm.normalize_observation(next_obs_np, update=True)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    print(f"Run dir: {run_dir}")
    print(
        f"Task: {task} | Obs dim: {obs_dim} | Action dim: {action_dim} | Device: {device} | "
        f"norm_obs={args.norm_obs} norm_reward={args.norm_reward} step_scale={step_scale}"
    )

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs * step_scale
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.view(-1)

            clipped_action = torch.clamp(action, action_low, action_high)
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, done_np, info = envs.step(clipped_action.cpu().numpy())
            train_reward_np = vecnorm.normalize_rewards(reward_np, done_np, update=True)
            next_obs_np = vecnorm.normalize_observation(next_obs_np, update=True)
            rewards[step] = torch.as_tensor(train_reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            finished = np.where(~np.isnan(info["episode_return"]))[0]
            for idx in finished:
                writer.add_scalar("charts/episodic_return", float(info["episode_return"][idx]), global_step)
                writer.add_scalar(
                    "charts/episodic_length",
                    float(info["episode_length"][idx]) * ep_length_scale,
                    global_step,
                )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        print(
            f"update={update:04d}/{num_updates} step={global_step} "
            f"policy_loss={pg_loss.item():.4f} value_loss={v_loss.item():.4f} "
            f"approx_kl={approx_kl.item():.5f} sps={sps}"
        )
        if args.save_every_updates > 0 and (update % args.save_every_updates == 0):
            ckpt_path = os.path.join(run_dir, f"ckpt_update_{update:06d}.pt")
            save_checkpoint(
                ckpt_path,
                agent=agent,
                optimizer=optimizer,
                vecnorm=vecnorm,
                args=args,
                global_step=global_step,
                update=update,
            )
            print(f"Saved checkpoint: {ckpt_path}")
    
    if args.save_model:
        model_path = os.path.join(run_dir, "final.pt")
        save_checkpoint(
            model_path,
            agent=agent,
            optimizer=optimizer,
            vecnorm=vecnorm,
            args=args,
            global_step=global_step,
            update=num_updates,
        )
        print(f"Model saved to {model_path}")

    if args.eval_episodes > 0:
        episodic_returns = evaluate(
            agent=agent,
            task=task,
            action_repeat=args.action_repeat,
            time_limit=args.time_limit,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            device=device,
            vecnorm=vecnorm,
            deterministic=args.eval_deterministic,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        eval_mean = float(np.mean(episodic_returns))
        eval_std = float(np.std(episodic_returns))
        print(f"Eval ({args.eval_episodes} eps): mean={eval_mean:.3f} std={eval_std:.3f}")


    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

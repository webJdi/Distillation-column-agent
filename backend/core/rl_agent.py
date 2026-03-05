"""
Deep RL Agent for CDU Optimization.

Uses Stable-Baselines3's SAC (Soft Actor-Critic) with optional
curriculum learning and custom callbacks for WebSocket progress updates.

Captures detailed metrics: critic/actor loss, mean Q-value, entropy,
alpha, action distribution, gradient norms, replay buffer size.
"""
from __future__ import annotations

import os
import time
import json
import threading
from datetime import datetime
from typing import Any, Callable, Optional
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.core.rl_environment import CDUEnvironment, ACTION_KEYS
from backend.models.schemas import (
    TrainingConfig,
    TrainingProgress,
    TrainingStatus,
    TrainingResult,
)
from backend.config import settings


# ── Algorithm registry ──────────────────────────────────────────────────────
ALGO_MAP = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}


def _safe_float(v):
    """Convert a possibly-stringified numeric value to float, or None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Custom callback for detailed live progress ────────────────────────────

class ProgressCallback(BaseCallback):
    """
    Captures comprehensive training metrics (losses, Q-values, entropy,
    alpha, action distribution, gradient norms, replay buffer) and pushes
    them to a callback function (typically WebSocket broadcast).
    """

    def __init__(
        self,
        broadcast_fn: Optional[Callable[[dict], None]] = None,
        log_interval: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.broadcast_fn = broadcast_fn
        self.log_interval = log_interval

        # Episode tracking
        self.episode_rewards: list[float] = []
        self.current_episode_reward = 0.0
        self.episode_count = 0
        self.best_reward = float("-inf")

        # Action tracking for distribution
        self._recent_actions: list[list[float]] = []

        # Full metrics history (kept for post-training analysis)
        self.metrics_history: list[dict] = []

    def _on_step(self) -> bool:
        # Accumulate reward
        reward = self.locals.get("rewards", [0])[0]
        self.current_episode_reward += reward

        # Track actions for distribution analysis
        action = self.locals.get("actions")
        if action is not None:
            a = action[0]
            self._recent_actions.append(
                a.tolist() if hasattr(a, "tolist") else list(a)
            )
            if len(self._recent_actions) > 1000:
                self._recent_actions = self._recent_actions[-1000:]

        # Check for episode end
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
            self.current_episode_reward = 0.0

        # Broadcast every N steps
        if self.num_timesteps % self.log_interval == 0 and self.broadcast_fn:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            try:
                self.broadcast_fn(metrics)
            except Exception as exc:
                logger.warning(f"Broadcast failed: {exc}")

        return True  # continue training

    # ── Detailed metric collection ──────────────────────────────────────

    @staticmethod
    def _to_python(val):
        """Convert numpy/torch scalars to native Python types for JSON."""
        if val is None:
            return None
        # Python builtins: pass through
        if isinstance(val, (bool, int, float, str)):
            # Clamp inf/nan for JSON compatibility
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                return 0.0
            return val
        # Any numpy scalar
        if isinstance(val, np.generic):
            v = val.item()
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                return 0.0
            return v
        # numpy array → list
        if isinstance(val, np.ndarray):
            return val.tolist()
        # dict: recurse
        if isinstance(val, dict):
            return {k: ProgressCallback._to_python(v) for k, v in val.items()}
        # list/tuple: recurse
        if isinstance(val, (list, tuple)):
            return [ProgressCallback._to_python(v) for v in val]
        # torch tensor
        if hasattr(val, "item"):
            try:
                return float(val.item())
            except Exception:
                pass
        return val

    def _collect_metrics(self) -> dict:
        """Collect all available training metrics from the SB3 model."""
        avg_rew = (
            float(np.mean(self.episode_rewards[-100:]))
            if self.episode_rewards
            else 0.0
        )

        metrics: dict[str, Any] = {
            "status": "training",
            "current_step": self.num_timesteps,
            "total_steps": self.locals.get("total_timesteps", 0),
            "episode": self.episode_count,
            "episode_reward": (
                self.episode_rewards[-1] if self.episode_rewards else 0.0
            ),
            "avg_reward": avg_rew,
            "best_reward": self.best_reward,
            "profit": avg_rew * 1000,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # ── SB3 logger metrics (critic/actor loss, ent_coef, etc.) ──────
        if hasattr(self.model, "logger") and hasattr(
            self.model.logger, "name_to_value"
        ):
            logged = self.model.logger.name_to_value
            metrics["critic_loss"] = float(logged.get("train/critic_loss", 0))
            metrics["actor_loss"] = float(logged.get("train/actor_loss", 0))
            metrics["ent_coef"] = float(logged.get("train/ent_coef", 0))
            metrics["ent_coef_loss"] = float(
                logged.get("train/ent_coef_loss", 0)
            )
            metrics["n_updates"] = int(logged.get("train/n_updates", 0))

        # ── Replay buffer ───────────────────────────────────────────────
        if (
            hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
        ):
            buf = self.model.replay_buffer
            metrics["replay_buffer_size"] = int(buf.size())
            metrics["replay_buffer_capacity"] = int(buf.buffer_size)
            metrics["replay_buffer_pct"] = round(
                buf.size() / buf.buffer_size * 100, 1
            )

        # ── Mean Q-value (sample a mini-batch from replay buffer) ───────
        try:
            if (
                hasattr(self.model, "critic")
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
                and self.model.replay_buffer.size() > self.model.batch_size
            ):
                replay_data = self.model.replay_buffer.sample(
                    min(64, self.model.replay_buffer.size())
                )
                with torch.no_grad():
                    q_values = self.model.critic(
                        replay_data.observations, replay_data.actions
                    )
                    # critic returns a tuple of Q tensors (Q1, Q2)
                    if isinstance(q_values, (list, tuple)):
                        all_q = torch.cat(
                            [q.flatten() for q in q_values], dim=0
                        )
                    else:
                        all_q = q_values.flatten()
                    metrics["mean_q_value"] = round(float(all_q.mean().item()), 4)
        except Exception:
            pass

        # ── Entropy (from actor log-prob on sampled observations) ───────
        try:
            if (
                hasattr(self.model, "actor")
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
                and self.model.replay_buffer.size() > self.model.batch_size
            ):
                replay_data = self.model.replay_buffer.sample(
                    min(64, self.model.replay_buffer.size())
                )
                with torch.no_grad():
                    actions_pi, log_prob = self.model.actor.action_log_prob(
                        replay_data.observations
                    )
                    metrics["entropy"] = round(
                        float(-log_prob.mean().item()), 4
                    )
        except Exception:
            pass

        # ── Gradient norms ──────────────────────────────────────────────
        try:
            for net_name, net_attr in [("actor", "actor"), ("critic", "critic")]:
                network = getattr(self.model, net_attr, None)
                if network is None:
                    continue
                total_norm = 0.0
                param_count = 0
                for p in network.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        param_count += 1
                if param_count > 0:
                    metrics[f"{net_name}_grad_norm"] = round(
                        float(total_norm**0.5), 6
                    )
        except Exception:
            pass

        # ── Action distribution ─────────────────────────────────────────
        if len(self._recent_actions) > 10:
            arr = np.array(self._recent_actions[-200:])
            metrics["action_distribution"] = {
                "means": np.round(arr.mean(axis=0), 4).tolist(),
                "stds": np.round(arr.std(axis=0), 4).tolist(),
                "names": list(ACTION_KEYS),
            }

        # Sanitize all numpy/torch types to Python natives
        return self._to_python(metrics)


# ── Agent manager ───────────────────────────────────────────────────────────

class RLAgentManager:
    """
    Manages training, inference, and checkpointing of an RL agent
    for the CDU environment.
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.env: Optional[CDUEnvironment] = None
        self.config: Optional[TrainingConfig] = None
        self.status = TrainingStatus.IDLE
        self._training_thread: Optional[threading.Thread] = None
        self._progress_callback: Optional[ProgressCallback] = None
        self._training_start_time: float = 0
        self._latest_progress: Optional[dict] = None
        self._latest_run_id: Optional[str] = None
        self._latest_metrics_history: list[dict] = []

        # Ensure checkpoint dir exists
        os.makedirs(settings.RL_CHECKPOINT_DIR, exist_ok=True)

    @property
    def is_training(self) -> bool:
        return self.status == TrainingStatus.TRAINING

    @property
    def latest_progress(self) -> Optional[dict]:
        return self._latest_progress

    @property
    def metrics_history(self) -> list[dict]:
        """Full metrics history for the latest training run."""
        return self._latest_metrics_history

    # ── Training ────────────────────────────────────────────────────────

    def start_training(
        self,
        config: TrainingConfig,
        prices: Optional[dict[str, float]] = None,
        broadcast_fn: Optional[Callable] = None,
        use_mock: bool = False,
    ) -> None:
        """Kick off training in a background thread."""
        if self.is_training:
            raise RuntimeError("Training already in progress")

        self.config = config
        self.status = TrainingStatus.TRAINING
        self._latest_metrics_history = []

        def _store_and_broadcast(data: dict):
            self._latest_progress = data
            if broadcast_fn:
                broadcast_fn(data)

        self._training_thread = threading.Thread(
            target=self._train_loop,
            args=(config, prices, _store_and_broadcast, use_mock),
            daemon=True,
        )
        self._training_thread.start()
        logger.info(f"Training started: {config.algorithm} for {config.total_timesteps} steps")

    def _train_loop(
        self,
        config: TrainingConfig,
        prices: Optional[dict[str, float]],
        broadcast_fn: Optional[Callable],
        use_mock: bool,
    ) -> None:
        """Background training loop with detailed metric capture."""
        try:
            self._training_start_time = time.time()

            # Create environment
            def make_env():
                env = CDUEnvironment(
                    prices=prices,
                    max_steps=200,
                    curriculum_level=1.0 if not config.use_curriculum else 0.3,
                    use_mock=use_mock,
                )
                return env

            vec_env = DummyVecEnv([make_env])
            self.env = vec_env.envs[0]

            # Create model
            AlgoClass = ALGO_MAP.get(config.algorithm, SAC)
            model_kwargs = {
                "policy": "MlpPolicy",
                "env": vec_env,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "gamma": config.gamma,
                "verbose": 0,
                "device": "auto",
            }

            # SAC-specific kwargs
            if config.algorithm == "SAC":
                model_kwargs["tau"] = settings.RL_TAU
                model_kwargs["buffer_size"] = settings.RL_BUFFER_SIZE
                model_kwargs["learning_starts"] = 1000

            self.model = AlgoClass(**model_kwargs)

            # Enhanced callback with detailed metrics
            self._progress_callback = ProgressCallback(
                broadcast_fn=broadcast_fn,
                log_interval=max(100, config.total_timesteps // 500),
            )

            # Curriculum learning: train in stages
            if config.use_curriculum:
                stages = [
                    (0.3, config.total_timesteps // 3),
                    (0.6, config.total_timesteps // 3),
                    (1.0, config.total_timesteps - 2 * (config.total_timesteps // 3)),
                ]
                for level, steps in stages:
                    self.env.curriculum_level = level
                    logger.info(f"Curriculum stage: level={level}, steps={steps}")
                    self.model.learn(
                        total_timesteps=steps,
                        callback=self._progress_callback,
                        reset_num_timesteps=False,
                    )
            else:
                self.model.learn(
                    total_timesteps=config.total_timesteps,
                    callback=self._progress_callback,
                )

            # Save checkpoint
            elapsed = time.time() - self._training_start_time
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"cdu_{config.algorithm}_{ts}"
            model_path = os.path.join(settings.RL_CHECKPOINT_DIR, run_id)
            self.model.save(model_path)

            # Store the metrics history
            self._latest_metrics_history = (
                self._progress_callback.metrics_history.copy()
            )
            self._latest_run_id = run_id

            self.status = TrainingStatus.COMPLETED
            logger.info(f"Training completed in {elapsed:.1f}s → {model_path}")

            # Build final summary with all metrics
            cb = self._progress_callback
            final_metrics = cb._collect_metrics() if cb else {}
            final_metrics.update({
                "status": "completed",
                "current_step": config.total_timesteps,
                "total_steps": config.total_timesteps,
                "episode": cb.episode_count,
                "episode_reward": (
                    cb.episode_rewards[-1] if cb.episode_rewards else 0.0
                ),
                "avg_reward": (
                    float(np.mean(cb.episode_rewards[-100:]))
                    if cb.episode_rewards
                    else 0.0
                ),
                "best_reward": float(cb.best_reward),
                "profit": float(cb.best_reward) * 1000,
                "training_time_seconds": round(elapsed, 2),
                "checkpoint_path": model_path,
                "checkpoint_size_mb": round(
                    os.path.getsize(model_path + ".zip") / 1024 / 1024, 2
                ),
                "run_id": run_id,
                "config": config.model_dump(),
                "timestamp": datetime.utcnow().isoformat(),
            })
            # Sanitize all values to native Python types for JSON
            final_metrics = ProgressCallback._to_python(final_metrics)

            # Save metrics alongside the checkpoint
            metrics_path = model_path + "_metrics.json"
            sanitized_history = ProgressCallback._to_python(
                self._latest_metrics_history
            )
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "config": config.model_dump(),
                        "final_metrics": final_metrics,
                        "metrics_history": sanitized_history,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Metrics saved → {metrics_path}")

            # Broadcast final state
            if broadcast_fn:
                broadcast_fn(final_metrics)

        except Exception as exc:
            self.status = TrainingStatus.ERROR
            logger.exception(f"Training failed: {exc}")
            if broadcast_fn:
                broadcast_fn({
                    "status": "error",
                    "error": str(exc),
                    "timestamp": datetime.utcnow().isoformat(),
                })

    # ── Inference ───────────────────────────────────────────────────────

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Get the agent's recommended action for an observation."""
        if self.model is None:
            raise RuntimeError("No model loaded — train or load a checkpoint first")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def load_checkpoint(self, path: str) -> None:
        """Load a previously saved model."""
        if not os.path.exists(path + ".zip") and not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        # Determine algo from filename
        algo_name = "SAC"
        for name in ALGO_MAP:
            if name.lower() in path.lower():
                algo_name = name
                break
        AlgoClass = ALGO_MAP[algo_name]
        self.model = AlgoClass.load(path)
        self.status = TrainingStatus.COMPLETED
        logger.info(f"Loaded checkpoint: {path} ({algo_name})")

        # Try to load associated metrics
        metrics_file = path + "_metrics.json"
        if not os.path.exists(metrics_file):
            metrics_file = path.replace(".zip", "") + "_metrics.json"
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                self._latest_metrics_history = data.get("metrics_history", [])
                self._latest_run_id = data.get("run_id")
                logger.info(f"Loaded metrics history ({len(self._latest_metrics_history)} points)")
            except Exception:
                pass

    def list_checkpoints(self) -> list[dict]:
        """List available model checkpoints with metrics summary."""
        checkpoints = []
        cp_dir = settings.RL_CHECKPOINT_DIR
        if os.path.exists(cp_dir):
            for f in sorted(os.listdir(cp_dir)):
                if f.endswith(".zip"):
                    fpath = os.path.join(cp_dir, f)
                    cp_info = {
                        "name": f.replace(".zip", ""),
                        "path": fpath,
                        "size_mb": round(os.path.getsize(fpath) / 1024 / 1024, 2),
                        "created": datetime.fromtimestamp(
                            os.path.getctime(fpath)
                        ).isoformat(),
                    }
                    # Attach metrics summary if available
                    metrics_file = fpath.replace(".zip", "_metrics.json")
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, "r") as mf:
                                mdata = json.load(mf)
                            fm = mdata.get("final_metrics", {})
                            cp_info["metrics_summary"] = {
                                "best_reward": _safe_float(fm.get("best_reward")),
                                "avg_reward": _safe_float(fm.get("avg_reward")),
                                "episodes": fm.get("episode"),
                                "training_time": _safe_float(fm.get("training_time_seconds")),
                                "critic_loss": _safe_float(fm.get("critic_loss")),
                                "actor_loss": _safe_float(fm.get("actor_loss")),
                                "mean_q_value": _safe_float(fm.get("mean_q_value")),
                            }
                        except Exception:
                            pass
                    checkpoints.append(cp_info)
        return checkpoints

    def get_run_metrics(self, run_id: Optional[str] = None) -> Optional[dict]:
        """Get the full metrics for a training run."""
        if run_id is None:
            run_id = self._latest_run_id
        if run_id is None:
            return None

        # Try to load from file
        metrics_file = os.path.join(
            settings.RL_CHECKPOINT_DIR, f"{run_id}_metrics.json"
        )
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f)

        # Return in-memory history if this is the current run
        if run_id == self._latest_run_id and self._latest_metrics_history:
            return {
                "run_id": run_id,
                "metrics_history": self._latest_metrics_history,
            }

        return None

    def stop_training(self) -> None:
        """Request training to stop (will finish current episode)."""
        self.status = TrainingStatus.IDLE
        logger.info("Training stop requested")

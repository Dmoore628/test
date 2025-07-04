"""
train.py

Purpose:
  This script sets up and trains a PPO reinforcement learning model using Stable Baselines3 on the 
  MNQ futures trading simulation environment (TradingDataEnv). The environment generates a 42-dimensional 
  observation vector that includes comprehensive market data, current trade metrics, and most recent closed 
  trade metrics. It also logs complete trade details for accurate P&L calculations and dashboard display.
  
  The training process supports transfer learning across trading days by attempting to load from a checkpoint 
  if available. Custom callbacks manage checkpoint saving, display a progress bar via tqdm, and (optionally) render 
  the environment in human mode.
  
Usage:
  Run this script with optional command-line arguments:
    --timesteps: Override the total number of training timesteps.
    --model_path: Provide a path to an existing model checkpoint to resume training.
    --experiment: Name of the experiment (used to organize logs and model checkpoints).
"""

import os
import argparse
import yaml
import logging
from time import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the trading environment
from trading_env import TradingDataEnv

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler('trading_training.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# -----------------------------------------------------------------------------
# Custom Callbacks
# -----------------------------------------------------------------------------
class ProgressBarCallback(BaseCallback):
    """
    Custom callback to display a progress bar during training using tqdm.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        from tqdm import tqdm
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", ncols=100)

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None

class RenderCallback(BaseCallback):
    """
    Custom callback that calls the environment's render() method after each step
    if the environment's render_mode is set to human.
    """
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.training_env.env_method("render")
        return True

# -----------------------------------------------------------------------------
# Utility Function: Load Configuration
# -----------------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    """
    Load configuration parameters from a YAML file.
    
    Args:
      config_path (str): The path to the configuration file.
      
    Returns:
      dict: The configuration parameters.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="RL Trading Model Training")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (overrides config)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to an existing model checkpoint to resume training")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Name of the experiment for organizing logs and checkpoints")
    args = parser.parse_args()

    # Load configuration
    config = load_config("config.yaml")
    total_timesteps = args.timesteps if args.timesteps is not None else config["ppo"]["total_timesteps"]

    # Determine experiment name and create required directories.
    experiment_name = args.experiment if args.experiment is not None else config["logging"]["experiment_name"]
    model_dir = os.path.join(config["logging"]["model_dir"], experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    eval_dir = os.path.join(config["logging"]["eval_dir"], experiment_name)
    os.makedirs(eval_dir, exist_ok=True)

    # Create the environment using a factory function
    def make_env():
        return TradingDataEnv(
            csv_path=config["data_path"],
            config_path="config.yaml",
            render_mode=config["environment"].get("render_mode", "fast")
        )
    env = DummyVecEnv([make_env])

    # Validate environment compliance with Gymnasium standards.
    try:
        check_env(env.envs[0], warn=True)
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        raise

    # Attempt to load a checkpoint for transfer learning.
    model_path = args.model_path
    default_model_path = os.path.join(model_dir, "final_ppo_model.zip")
    model = None
    if model_path is None and os.path.exists(default_model_path):
        model_path = default_model_path
        logger.info(f"Resuming training from model: {default_model_path}")
    if model_path is not None:
        try:
            logger.info(f"Loading model from {model_path}")
            model = PPO.load(model_path, env=env)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}. Initializing new model.")
            model = None

    # If no model is loaded, initialize a new PPO model.
    if model is None:
        logger.info("Initializing new PPO model...")
        ppo_params = config["ppo"]
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=ppo_params["learning_rate"],
            n_steps=ppo_params["n_steps"],
            batch_size=ppo_params["batch_size"],
            gamma=ppo_params["gamma"],
            gae_lambda=ppo_params["gae_lambda"],
            clip_range=ppo_params["clip_range"],
            ent_coef=ppo_params["ent_coef"],
            vf_coef=ppo_params["vf_coef"],
            max_grad_norm=ppo_params["max_grad_norm"],
        )

    # Set up callbacks: checkpoint saving, progress bar, and optional rendering.
    checkpoint_callback = CheckpointCallback(
        save_freq=config["logging"]["save_model_freq"],
        save_path=model_dir,
        name_prefix="ppo_model"
    )
    callbacks = [checkpoint_callback, ProgressBarCallback(total_timesteps)]
    if config["environment"].get("render_mode", "fast") == "human":
        callbacks.append(RenderCallback())

    # Start training.
    logger.info("Starting training...")
    start_time = time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    end_time = time()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    # Save the final model.
    final_model_path = os.path.join(model_dir, "final_ppo_model.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()

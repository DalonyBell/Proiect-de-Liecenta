from stable_baselines3 import PPO
import os
from environment import CarEnv
import time

print('This is the start of the training script')

print('Setting folders for logs and models')
models_dir = f"models-{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print('Connecting to the environment...')
env = CarEnv()

env.reset()
print('Environment has been reset as part of the launch')

model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001, tensorboard_log=logdir)

TIMESTEPS = 500_000  # Number of timesteps per training iteration
TOTAL_TIMESTEPS = 2_000_000  # Total number of timesteps to train
iters = 0

print(f'Starting training for a total of {TOTAL_TIMESTEPS} timesteps.')
while model.num_timesteps < TOTAL_TIMESTEPS:
    iters += 1
    print(f'Iteration {iters} is about to commence...')
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_{iters}")
    print(f'Iteration {iters} has been trained')
    model.save(f"{models_dir}/{TIMESTEPS * iters}")

print('Training complete.')

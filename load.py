import gym
from stable_baselines3 import PPO, A2C
import os

from torch import tensor 



models_dir = "models/PPO"
generation = "290000" + ".zip"
model_path = f"{models_dir}/{generation}"


env = gym.make("CartPole-v1")
env.reset()

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
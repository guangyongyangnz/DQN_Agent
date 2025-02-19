import gym
from stable_baselines3 import DQN
# from stable_baselines3.common.vec_env import DummyVecEnv

# env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

env = gym.make("CartPole-v1", render_mode="human")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_cartpole_log/")
model.learn(total_timesteps=50000)

model.save("dqn_cartpole")

del model
model = DQN.load("dqn_cartpole")

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()

env.close()

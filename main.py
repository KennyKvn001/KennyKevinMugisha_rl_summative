from custom_env.city_env import CityEnv
import time

env = CityEnv(render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(1)

env.close()

from envs.dmc import DeepMindControl
import matplotlib.pyplot as plt
import numpy as np

env = DeepMindControl(
    name="cheetah_run",
    action_repeat=2,
    size=(480, 640),
    camera=0
)

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

obs = env.reset()
im = ax.imshow(obs['image'])

for _ in range(1000):  # Run for many steps
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Update the image
    im.set_data(obs['image'])
    plt.pause(0.01)  # Small pause to allow rendering
    
    if done:
        obs = env.reset()

plt.ioff()
plt.show()
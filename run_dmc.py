from envs.dmc import DeepMindControl
import numpy as np
import imageio

# Load using the custom wrapper
env = DeepMindControl(
    name="reacher_easy",
    action_repeat=2,
    size=(480, 640),  # or (480, 640) for higher resolution
    camera=0
)

# Collect frames
frames = []
obs = env.reset()

for _ in range(100):  # Run for 100 steps
    # The 'image' is already in the observation
    frames.append(obs['image'])
    
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        break

# Save video
imageio.mimsave('reacher_episode.mp4', frames, fps=30)
print("Video saved as reacher_episode.mp4")
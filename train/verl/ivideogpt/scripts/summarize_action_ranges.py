import glob
import os
import numpy as np
from tqdm import tqdm
import torch

root_path = '/home/NAS/rl_data/frame_action_datasets/fractal20220817_data/'
files = glob.glob(os.path.join(root_path, '*.npz'))

max_actions = np.ones((13, ), dtype=np.float32) * -1e5
min_actions = np.ones((13, ), dtype=np.float32) * 1e5

for file in tqdm(files):
    data = np.load(file)
    actions = data['action']
    max_actions = np.maximum(max_actions, actions.max(axis=0))
    min_actions = np.minimum(min_actions, actions.min(axis=0))

action_ranges = np.stack([min_actions, max_actions], axis=1)
action_ranges = torch.from_numpy(action_ranges).float()
torch.save(action_ranges, 'ivideogpt/configs/frac_action_ranges.pth')
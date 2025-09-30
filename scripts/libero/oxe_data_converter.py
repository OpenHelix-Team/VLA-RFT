import numpy as np
import os
from tqdm import tqdm
import argparse
import tensorflow_datasets as tfds


DISPLAY_KEY = {
    'taco_play': 'rgb_static',
    'roboturk': 'front_rgb',
    'viola': 'agentview_rgb',
    'language_table': 'rgb',
    'stanford_robocook_converted_externally_to_rlds1': 'image_1',
    'stanford_robocook_converted_externally_to_rlds2': 'image_2',
    'stanford_robocook_converted_externally_to_rlds3': 'image_3',
    'stanford_robocook_converted_externally_to_rlds4': 'image_4',
    'uiuc_d3field1': 'image_1',
    'uiuc_d3field2': 'image_2',
    'uiuc_d3field3': 'image_3',
    'uiuc_d3field4': 'image_4',
}


def get_dataset_path(parent_dir, dataset_name):
    if dataset_name == 'robo_net' or dataset_name == 'cmu_playing_with_food':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    elif dataset_name[:-1] == 'uiuc_d3field' or dataset_name[:-1] == 'stanford_robocook_converted_externally_to_rlds':
        dataset_name = dataset_name[:-1]
        version = '0.1.0'
    else:
        version = '1.0.0'
    return os.path.join(parent_dir, dataset_name, version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='fractal20220817_data')
    parser.add_argument('--input_path', type=str, default='/data2/tensorflow_datasets')
    parser.add_argument('--output_path', type=str, default='/data2/frame_action_datasets')
    parser.add_argument('--max_num_episodes', default=None, type=int)
    parser.add_argument('--begin_num_episodes', default=0, type=int)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    ds = tfds.builder_from_directory(builder_dir=get_dataset_path(args.input_path, dataset_name)).as_dataset()
    display_key = DISPLAY_KEY.get(dataset_name, 'image')
    root_path = os.path.join(args.output_path)
    os.makedirs(root_path, exist_ok=True)

    num_episodes = args.begin_num_episodes
    for key in ds:
        bar = tqdm(enumerate(ds[key]))
        for i, episode in bar:
            if os.path.exists(os.path.join(root_path, f'{key}_eps_{num_episodes:08d}.npz')):
                continue
            frames, actions, language_instruction, states= [], [], [], []
            for step in episode['steps']:
                # print("Step keys:", list(step.keys()))
                frames.append(step['observation'][display_key])
                states.append(step['observation']['state'])
                if callable(getattr(step['action'], 'keys', None)):
                    actions.append(np.concatenate([step['action'][k] for k in step['action'].keys()]))
                else:
                    actions.append(step['action'])
                language_instruction.append(step['language_instruction'])
                # breakpoint()
            frames = np.array(frames)
            actions = np.array(actions)
            language_instruction = np.array(language_instruction)
            bar.set_postfix(epslen=len(frames))
            np.savez_compressed(os.path.join(root_path, f'{key}_eps_{num_episodes:08d}.npz'), **{display_key: frames, 'action': actions, 'lang': language_instruction, 'state': states})
            num_episodes += 1
            if args.max_num_episodes is not None and num_episodes >= args.max_num_episodes:
                break
    print(f"Total episodes processed: {num_episodes}")
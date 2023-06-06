

num_workers=4
import numpy as np

import concurrent.futures

def sample_indices_single(indices_save,importance,replace):

    indices_save = np.random.choice(importance.shape[0], size=indices_save.shape[0], p=importance,replace=replace)

    return None


def sample_indices_multi(indices_save, importance, replace):
    indices_save = np.random.choice(importance.shape[0], size=indices_save.shape[0], p=importance, replace=replace)

    indices_s

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future = executor.submit(
            sample_indices_single,
            all_imgs[current_index],
            video_path,
            img_wh,
            downsample,
            transform,
        )
        futures.append(future)
        current_index += 1



import requests
import numpy as np


def download_file_requests(url, local_filename):
    """
    Downloads a file from a given URL using the requests library.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The desired local filename to save the downloaded file as.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"File '{local_filename}' downloaded successfully from '{url}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def get_loss(bayer1, bayer2):
    return ((bayer1 - bayer2) ** 2).mean()


def check_if_crop_is_valid(shape, crop_edges):
    if (crop_edges[:2] < 0).any() or (crop_edges[:2] > shape[0]).any():
        return False
    if (crop_edges[-2:] < 0).any() or (crop_edges[-2:] > shape[1]).any():
        return False
    return True


def align_images(
    rh1, rh2, dims, offset=(0, 0, 0, 0), max_iters=100, step_sizes=[16, 8, 4, 2]
):
    offset = np.array(offset)
    bayer1 = rh1.input_handler(dims=dims)
    img_shape = rh1.raw.shape[-2:]

    loss = get_loss(bayer1, rh2.input_handler(dims=dims + offset))

    for step_size in step_sizes:
        directions = [
            np.array((step_size, step_size, 0, 0)),
            np.array((-step_size, -step_size, 0, 0)),
            np.array((0, 0, step_size, step_size)),
            np.array((0, 0, -step_size, -step_size)),
        ]
        for _ in range(max_iters):
            starting_offset = offset.copy()
            for step_dir in directions:
                crop_edges = dims + offset + step_dir
                # Do not update if step would create an invalid crop
                if not check_if_crop_is_valid(img_shape, crop_edges):
                    continue
                temp_loss = get_loss(bayer1, rh2.input_handler(dims=crop_edges))
                if temp_loss < loss:
                    offset += step_dir
                    loss = temp_loss
            if np.all(starting_offset == offset):
                break  # No improvement for this step size
    return offset

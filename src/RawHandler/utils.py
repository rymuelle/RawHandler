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


def transform_to_rggb(transform):
    """
    Transforms 3x3 color space transform to work with rggb color spaces.
    Args:
        transform (np.array): 3x3 numpy array that defines the colorspace transform.

    Returns:
        new_transform (np.array): 4x4 array for rggb data.
    """
    t = transform

    t00, t01, t02 = t[0, 0], t[0, 1], t[0, 2]
    t10, t11, t12 = t[1, 0], t[1, 1], t[1, 2]
    t20, t21, t22 = t[2, 0], t[2, 1], t[2, 2]

    new_transform = np.block(
        [
            [t00, t01 / 2, t01 / 2, t02],
            [t10, t11, 0.0, t12],
            [t10, 0.0, t11, t12],
            [t20, t21 / 2, t21 / 2, t22],
        ]
    )
    return new_transform


def make_colorspace_matrix(
    rgb_to_xyz, colorspace="lin_rec2020", xyz_to_colorspace=None
):
    """
    Computes the combination of the rgb to xyz converstion, and a convertion from xyz to the specified colorspace.
    Args:
        xyz_to_colorspace (np.array): Specify your own 3x3 matrix to convert to a colorspace. This arguement gets overwritten by the 'colorspace' arguement. (Optional)
        colorspace (str): Name of predefined colorspace: 'sRGB', 'AdobeRGB', 'lin_rec2020'. (Default 'lin_rec2020')
    Returns:
        transform (np.array): 3x3 array for rggb data.
    """
    if colorspace == "sRGB":
        xyz_to_colorspace = [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    elif colorspace == "AdobeRGB":
        xyz_to_colorspace = [
            [2.0413690, -0.5649464, -0.3446944],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0134474, -0.1183897, 1.0154096],
        ]
    elif colorspace == "lin_rec2020":
        xyz_to_colorspace = [
            [1.71666343, -0.35567332, -0.25336809],
            [-0.66667384, 1.61645574, 0.0157683],
            [0.01764248, -0.04277698, 0.94224328],
        ]
        xyz_to_colorspace = np.linalg.inv(xyz_to_colorspace)
    assert xyz_to_colorspace is not None, (
        "Color space not supported, please supply color space."
    )
    transform = xyz_to_colorspace @ rgb_to_xyz
    return transform

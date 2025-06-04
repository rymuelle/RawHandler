import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import rawpy

from RawHandler.utils import make_colorspace_matrix, transform_to_rggb


class RawHandler:
    """
    Wraps rawpy to make transforming raw data into useful formats for machine learning easy.

    Args:
        path (string): Path to raw file.
    """

    def __init__(self, path):
        self.path = path
        self.rawpy = rawpy.imread(path)
        self.raw = self.rawpy.raw_image_visible
        assert self.rawpy.color_desc.decode() == "RGBG", (
            "Only raw files with Bayer patters are supported currently."
        )
        self.bayer_pattern = self.bayer_pattern_description()

    def bayer_pattern_description(self):
        return "".join(map(lambda idx: "RGBG"[idx], self.rawpy.raw_pattern.flatten()))

    def remove_masked_pixels(self, img):
        return img[:, 0 : self.rawpy.sizes.iheight, 0 : self.rawpy.sizes.iwidth]

    def input_handler(self, dims=None, img=None):
        """
        Handles optional image and crop data.

        Args:
            image (np.array): Array of structure [C, H, W]. If 'None' returns the raw bayer data with masked pixels trimmed off. (optional)
            dims (int): Specify dimensions to crop. (Optional)
        Returns :
            img (np.array): Array of dimensions [C, H, W]
        """
        if img is None:
            img = np.expand_dims(self.raw, axis=0)
            img = self.remove_masked_pixels(img)
        elif len(img.shape) == 2:
            img = np.expand_dims(self.raw, axis=0)
        if dims is not None:
            if len(dims) != 4:
                raise ValueError(
                    f"Arguments must be length 0 or 4, found length {dims}."
                )
            # Center on Bayer grid
            h1, h2, w1, w2 = dims
            h1, h2, w1, w2 = list(map(lambda x: x - x % 2, [h1, h2, w1, w2]))
            return img[:, h1:h2, w1:w2]
        else:
            return img

    def make_bayer_map(self, bayer):
        channel_map = np.zeros_like(bayer)
        channel_map[0, 0::2, 0::2] = self.rawpy.raw_pattern[0][0]
        channel_map[0, 0::2, 1::2] = self.rawpy.raw_pattern[0][1]
        channel_map[0, 1::2, 0::2] = self.rawpy.raw_pattern[1][0]
        channel_map[0, 1::2, 1::2] = self.rawpy.raw_pattern[1][1]
        return channel_map

    def adjust_bayer_bw_levels(self, img=None, dims=None):
        img = self.input_handler(img=img, dims=dims)
        img = img.astype(np.float32)

        bayer_map = self.make_bayer_map(img)
        for channel in range(4):
            img[bayer_map == channel] -= self.rawpy.black_level_per_channel[channel]
            img[bayer_map == channel] *= 1.0 / (
                self.rawpy.white_level - self.rawpy.black_level_per_channel[channel]
            )
        return img

    def adjust_bayer_black_levels(self, bayer):
        bayer_map = self.make_bayer_map(bayer)
        for channel in range(4):
            bayer[bayer_map == channel] -= self.rawpy.black_level_per_channel[channel]
        return bayer

    def as_rggb(self, dims=None, img=None):
        """
        Stacks bayer data into a 4 channel image with half the dimensions.

        Args:
            image (np.array): Array of structure [C, H, W]. If 'None' returns the raw bayer data with masked pixels trimmed off. (optional)
            dims (int): Specify dimensions to crop. (Optional)
        Returns :
            rggb (np.array): Cropped and stacked version of the underlying raw data with dimenions [4, H / 2, W / 2].
        """
        raw = self.input_handler(dims=dims, img=img)
        raw = self.adjust_bayer_bw_levels(raw)

        def get_matching_index(channel):
            return [
                idx
                for idx in range(len(self.bayer_pattern))
                if self.bayer_pattern[idx] == channel
            ]

        r_idx = get_matching_index("R")
        g_idx = get_matching_index("G")
        b_idx = get_matching_index("B")
        assert (len(r_idx) == 1) and (len(g_idx) == 2) and (len(b_idx) == 1), (
            "Incorrect number of channels found."
        )
        smart_indexing_array = ((0, 0), (0, 1), (1, 0), (1, 1))
        rggb = np.stack(
            [raw[0, idxs[1] :: 2, idxs[0] :: 2] for idxs in smart_indexing_array]
        )
        return rggb

    def as_rgb(self, dims=None, img=None):
        """
        Demosaics the underlying bayer data into 3 channel RGB data without color spaces applied.

        Args:
            image (np.array): Array of structure [C, H, W]. If 'None' returns the raw bayer data with masked pixels trimmed off. (optional)
            dims (int): Specify dimensions to crop. (Optional)
        Returns:
            rgb (np.array): Cropped, demosaiced data with dimenions [3, H, W ].
        """
        assert self.bayer_pattern in ["RGGB", "BGGR", "GRBG", "GBRG"], (
            f"{self.bayer_pattern} is not supported for demosaicing."
        )
        raw = self.input_handler(dims=dims, img=img)
        raw = self.adjust_bayer_bw_levels(raw)
        rgb = demosaicing_CFA_Bayer_Menon2007(
            raw.transpose(1, 2, 0), pattern=self.bayer_pattern
        )
        return rgb.transpose(2, 0, 1)

    def rgb_colorspace_transform(self, **kwargs):
        rgb_to_xyz = np.linalg.inv(self.rawpy.rgb_xyz_matrix[:3])
        transform = make_colorspace_matrix(rgb_to_xyz, **kwargs)
        return transform

    def as_rgb_colorspace(
        self, dims=None, img=None, colorspace="lin_rec2020", xyz_to_colorspace=None
    ):
        """
        Converts or returns demosaiced data converted into specified colorspace.
        Resources: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

        Args:
            image (np.array): Array of structure [C, H, W]. If 'None' returns the raw bayer data with masked pixels trimmed off. (optional)
            dims (int): Specify dimensions to crop. (Optional)
            xyz_to_colorspace (np.array): Specify your own 3x3 matrix to convert to a colorspace. This arguement gets overwritten by the 'colorspace' arguement. (Optional)
            colorspace (str): Name of predefined colorspace: 'sRGB', 'AdobeRGB', 'lin_rec2020'. (Default 'lin_rec2020')
        Returns:
            rgb (np.array): Cropped, demosaiced, and profiled data with dimenions [3, H, W].
        """
        if img is None:
            img = self.as_rgb(dims=dims)
        else:
            img = self.input_handler(dims=dims, img=img)
        transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        orig_dims = img.shape
        return (transform @ img.reshape(3, -1)).reshape(orig_dims)

    def as_rggb_colorspace(
        self, dims=None, img=None, colorspace="lin_rec2020", xyz_to_colorspace=None
    ):
        """
        Converts or returns rggb data converted into specified colorspace.

        Args:
            image (np.array): Array of structure [C, H, W]. If 'None' returns the raw bayer data with masked pixels trimmed off. (optional)
            dims (int): Specify dimensions to crop. (Optional)
            xyz_to_colorspace (np.array): Specify your own 3x3 matrix to convert to a colorspace. This arguement gets overwritten by the 'colorspace' arguement. (Optional)
            colorspace (str): Name of predefined colorspace: 'sRGB', 'AdobeRGB', 'lin_rec2020'. (Default 'lin_rec2020')
        Returns:
            rggb (np.array): Cropped, demosaiced, and profiled data with dimenions [4, H // 2, W // 2].
        """
        if img is None:
            img = self.as_rggb(dims=dims)
        else:
            img = self.input_handler(dims=dims, img=img)
        transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        rggb_transform = transform_to_rggb(transform)
        print(rggb_transform)
        print(transform)
        orig_dims = img.shape
        return (rggb_transform @ img.reshape(4, -1)).reshape(orig_dims)

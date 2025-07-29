import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_bilinear
import rawpy
from typing import Dict, Any, NamedTuple, TypedDict, Optional
import datetime
import exifread # Import exifread
from RawHandler.utils import get_exif_data

from RawHandler.utils import make_colorspace_matrix, transform_to_rggb, pixel_unshuffle, pixel_shuffle

# Define a NamedTuple for the core metadata required by BaseRawHandler for processing
class CoreRawMetadata(NamedTuple):
    black_level_per_channel: np.ndarray
    white_level: int
    rgb_xyz_matrix: np.ndarray
    raw_pattern: np.ndarray
    iheight: int
    iwidth: int

# # Define a TypedDict for the additional, general metadata
# class FullRawMetadata(TypedDict, total=False):
#     make: Optional[str]
#     model: Optional[str]
#     artist: Optional[str]
#     description: Optional[str]
#     copyright: Optional[str]
#     datetime: Optional[datetime.datetime]
#     iso_speed: Optional[int]
#     exposure_time: Optional[float]
#     aperture: Optional[float] # FNumber
#     focal_length: Optional[float]
#     lens_make: Optional[str]
#     lens_model: Optional[str]
#     orientation: Optional[int]
#     shutter_speed_value: Optional[float] # APEX value
#     aperture_value: Optional[float] # APEX value
#     exposure_bias_value: Optional[float]
#     metering_mode: Optional[int] # exifread gives int for this
#     flash: Optional[int] # exifread gives int for this
#     color_space: Optional[int] # exifread gives int for this
#     image_width: Optional[int] # from exifread
#     image_height: Optional[int] # from exifread
#     sensing_method: Optional[int] # exifread gives int for this
#     # Add more fields as needed based on exifread output
#     software: Optional[str]
#     serial_number: Optional[str] # exifread can sometimes get this from MakerNotes

class BaseRawHandler:
    """
    Base class for handling raw image pixel data.

    Args:
        pixel_array (np.array): A 2D NumPy array representing the raw pixel data.
        core_metadata (CoreRawMetadata): A NamedTuple containing essential metadata for processing.
        full_metadata (Optional[FullRawMetadata]): A TypedDict containing additional, general metadata.
    """

    def __init__(self, pixel_array: np.ndarray, core_metadata: CoreRawMetadata, full_metadata: Optional[dict] = None):
        if not isinstance(pixel_array, np.ndarray):
            raise TypeError("pixel_array must be a NumPy array.")
        if not isinstance(core_metadata, CoreRawMetadata):
            raise TypeError("core_metadata must be an instance of CoreRawMetadata.")

        self.raw = pixel_array
        self.core_metadata = core_metadata
        self.full_metadata = full_metadata if full_metadata is not None else {}

    def _remove_masked_pixels(self, img: np.ndarray) -> np.ndarray:
        """Removes masked pixels from the image based on core_metadata.iheight and core_metadata.iwidth."""
        return img[:, 0 : self.core_metadata.iheight, 0 : self.core_metadata.iwidth]

    def _input_handler(self, dims=None, img: np.ndarray = None) -> np.ndarray:
        """
        Handles optional image and crop data.
        """
        if img is None:
            img = np.expand_dims(self.raw, axis=0)
            img = self._remove_masked_pixels(img)
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
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

    def _make_bayer_map(self, bayer: np.ndarray) -> np.ndarray:
        """Creates a Bayer channel map."""
        channel_map = np.zeros_like(bayer, dtype=int)
        channel_map[0, 0::2, 0::2] = 0  # Red
        channel_map[0, 0::2, 1::2] = 1  # Green (G1)
        channel_map[0, 1::2, 0::2] = 3  # Green (G2)
        channel_map[0, 1::2, 1::2] = 2  # Blue
        return channel_map

    def adjust_bayer_bw_levels(self, img: np.ndarray = None, dims=None) -> np.ndarray:
        """
        Adjusts black and white levels of Bayer data.
        """
        img = self._input_handler(img=img, dims=dims)
        img = img.astype(np.float32)

        bayer_map = self._make_bayer_map(img)
        for channel in range(4):
            img[bayer_map == channel] -= self.core_metadata.black_level_per_channel[channel]
            img[img<0] = 0
            img[bayer_map == channel] *= 1.0 / (
                self.core_metadata.white_level - self.core_metadata.black_level_per_channel[channel]
            )
        return img

    def adjust_bayer_black_levels(self, bayer: np.ndarray) -> np.ndarray:
        """
        Adjusts only the black levels of Bayer data.
        """
        bayer_map = self._make_bayer_map(bayer)
        for channel in range(4):
            bayer[bayer_map == channel] -= self.core_metadata.black_level_per_channel[channel]
        return bayer

    def as_rggb(self, dims=None, img: np.ndarray = None) -> np.ndarray:
        """
        Stacks bayer data into a 4 channel image with half the dimensions.
        """
        raw = self._input_handler(dims=dims, img=img)
        raw = self.adjust_bayer_bw_levels(raw)

        four_channel = pixel_unshuffle(raw, 2)
        return four_channel

    def as_rgb(self, dims=None, img: np.ndarray = None) -> np.ndarray:
        """
        Demosaics the underlying bayer data into 3 channel RGB data without color spaces applied.
        """
        raw = self._input_handler(dims=dims, img=img)
        raw = self.adjust_bayer_bw_levels(raw)
        print(raw.min(), raw.max())
        pattern = "".join(map(lambda idx: "RGBG"[idx], self.core_metadata.raw_pattern.flatten()))
        rgb = demosaicing_CFA_Bayer_bilinear(
            raw.transpose(1, 2, 0), pattern=pattern
        )
        return rgb.transpose(2, 0, 1)

    def rgb_colorspace_transform(self, **kwargs) -> np.ndarray:
        """
        Generates a color space transformation matrix.
        """
        rgb_to_xyz = np.linalg.inv(self.core_metadata.rgb_xyz_matrix[:3])
        transform = make_colorspace_matrix(rgb_to_xyz, **kwargs)
        return transform

    def as_rgb_colorspace(
        self, dims=None, img: np.ndarray = None, colorspace="lin_rec2020", xyz_to_colorspace: np.ndarray = None
    ) -> np.ndarray:
        """
        Converts or returns demosaiced data converted into specified colorspace.
        """
        if img is None:
            img = self.as_rgb(dims=dims)
        else:
            img = self._input_handler(dims=dims, img=img)
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = self.adjust_bayer_bw_levels(img)
                #pattern = "".join(map(lambda idx: "RGBG"[idx], self.core_metadata.raw_pattern.flatten()))
                img = demosaicing_CFA_Bayer_bilinear(img.transpose(1, 2, 0)).transpose(2,0,1)

        transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        orig_dims = img.shape
        return (transform @ img.reshape(3, -1)).reshape(orig_dims)

    def as_rggb_colorspace(
        self, dims=None, img: np.ndarray = None, colorspace="lin_rec2020", xyz_to_colorspace: np.ndarray = None
    ) -> np.ndarray:
        """
        Converts or returns rggb data converted into specified colorspace.
        """
        if img is None:
            img = self.as_rggb(dims=dims)
        else:
            img = self._input_handler(dims=dims, img=img)
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = self.as_rggb(dims=dims, img=img)

        transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        rggb_transform = transform_to_rggb(transform)
        orig_dims = img.shape
        return (rggb_transform @ img.reshape(4, -1)).reshape(orig_dims)


    def downsize(self, min_preview_size=256):
        H, W = self.raw.shape
        W_steps, H_steps = H // min_preview_size - 1, W // min_preview_size - 1
        steps = min(W_steps, H_steps)
        rggb = pixel_unshuffle(np.expand_dims(self.raw, 0), 2)[:, ::steps, ::steps]
        mosaic = pixel_shuffle(rggb, 2)
        return mosaic

    def generate_thumbnail(self, min_preview_size=256, colorspace="sRGB"):
        img = self.downsize(min_preview_size=min_preview_size)
        img = self.as_rgb_colorspace(img=img, colorspace=colorspace)
        return img

class RawHandler:
    """
    Factory class to create BaseRawHandler instances from raw image files.
    This class handles rawpy specific parsing for pixel data and core metadata,
    and uses exifread for extracting general EXIF metadata.

    Args:
        path (string): Path to raw file.
    """

    def __new__(cls, path: str):
        # Use rawpy for raw pixel data and core processing metadata
        rawpy_object = rawpy.imread(path)
        raw_image = rawpy_object.raw_image_visible

        assert rawpy_object.color_desc.decode() == "RGBG", (
            "Only raw files with Bayer patterns are supported currently."
        )

        bayer_pattern = "".join(map(lambda idx: "RGBG"[idx], rawpy_object.raw_pattern.flatten()))

        # Adjust raw_image based on Bayer pattern to align with RGGB
        if bayer_pattern == "RGGB":
            pass
        elif bayer_pattern == "BGGR":
            raw_image = raw_image[1:-1, 1:-1]
        elif bayer_pattern == "GBRG":
            raw_image = raw_image[1:-1, :]
        elif bayer_pattern == "GRBG":
            raw_image = raw_image[:, 1:-1]
        else:
            print(f'{bayer_pattern} not supported.')
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

        # 1. Extract Core Metadata for BaseRawHandler's processing logic
        core_metadata = CoreRawMetadata(
            black_level_per_channel=rawpy_object.black_level_per_channel,
            white_level=rawpy_object.white_level,
            rgb_xyz_matrix=rawpy_object.rgb_xyz_matrix,
            raw_pattern=rawpy_object.raw_pattern,
            iheight=rawpy_object.sizes.iheight,
            iwidth=rawpy_object.sizes.iwidth,
        )

        # 2. Extract Full (General) Metadata using exifread
        metadata = get_exif_data(path)

        return BaseRawHandler(pixel_array=raw_image, core_metadata=core_metadata, full_metadata=metadata)
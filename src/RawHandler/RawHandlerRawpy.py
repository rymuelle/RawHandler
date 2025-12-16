import numpy as np
import rawpy
from typing import NamedTuple, Optional
from RawHandler.utils import get_exif_data, sparse_representation_three_channel
from typing import Literal

from RawHandler.utils import (
    make_colorspace_matrix,
    pixel_unshuffle,
)


# Define a NamedTuple for the core metadata required by BaseRawHandler for processing
class CoreRawMetadata(NamedTuple):
    black_level_per_channel: np.ndarray
    white_level: int
    rgb_xyz_matrix: np.ndarray
    raw_pattern: np.ndarray
    camera_white_balance: np.ndarray
    iheight: int
    iwidth: int


class BaseRawHandlerRawpy:
    """
    Base class for handling raw image pixel data.

    Args:
        pixel_array (np.array): A 2D NumPy array representing the raw pixel data.
        core_metadata (CoreRawMetadata): A NamedTuple containing essential metadata for processing.
        full_metadata (Optional[FullRawMetadata]): A Dict containing additional, general metadata.
    """

    def __init__(
        self,
        rawpy_object: rawpy.RawPy,
        core_metadata: CoreRawMetadata,
        full_metadata: Optional[dict] = None,
        colorspace: Literal[
            "camera", "XYZ", "sRGB", "AdobeRGB", "lin_rec2020"
        ] = "lin_rec2020",
    ):
        if not isinstance(core_metadata, CoreRawMetadata):
            raise TypeError("core_metadata must be an instance of CoreRawMetadata.")

        self.rawpy_object = rawpy_object
        self.core_metadata = core_metadata
        self.full_metadata = full_metadata if full_metadata is not None else {}
        self.colorspace = colorspace
        self.xyz_linear = None

    def compute_linear(self):
        xyz_linear = (
            self.rawpy_object.postprocess(
                output_color=rawpy.ColorSpace.XYZ,
                no_auto_bright=True,
                use_camera_wb=False,
                use_auto_wb=False,
                gamma=(1, 1),
                user_flip=0,
                output_bps=16,
            )
            / 65535
        )
        self.xyz_linear = xyz_linear.transpose(2, 0, 1)

    def _input_handler(self, dims=None, safe_crop=0) -> np.ndarray:
        """
        Crops linear array.
        """
        if self.xyz_linear is None:
            self.compute_linear()
        if dims is not None:
            h1, h2, w1, w2 = dims
            if safe_crop:
                h1, h2, w1, w2 = list(
                    map(lambda x: x - x % safe_crop, [h1, h2, w1, w2])
                )
            return self.xyz_linear[:, h1:h2, w1:w2]
        else:
            return self.xyz_linear

    def rgb_colorspace_transform(self, colorspace=None, **kwargs) -> np.ndarray:
        """
        Generates a color space transformation matrix for this image.
        """
        colorspace = colorspace or self.colorspace

        rgb_to_xyz = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        if colorspace == "XYZ":
            return rgb_to_xyz

        transform = make_colorspace_matrix(rgb_to_xyz, colorspace=colorspace, **kwargs)
        return transform

    def apply_colorspace_transform(
        self,
        dims=None,
        safe_crop=0,
        xyz_to_colorspace: np.ndarray = None,
        colorspace=None,
        clip=False,
    ) -> np.ndarray:
        """
        Converts or returns linear data converted into specified colorspace.
        """
        xyz_linear = self._input_handler(dims=dims, safe_crop=safe_crop)
        rgb_transform = self.rgb_colorspace_transform(
            colorspace=colorspace, xyz_to_colorspace=xyz_to_colorspace
        )
        orig_dims = xyz_linear.shape
        transformed = (rgb_transform @ xyz_linear.reshape(3, -1)).reshape(orig_dims)
        if clip:
            transformed = np.clip(transformed, 0, 1)
        return transformed

    def downsize(
        self, min_preview_size=256, colorspace=None, clip=False, safe_crop=0
    ) -> np.ndarray:
        _, H, W = self.xyz_linear.shape
        W_steps, H_steps = H // min_preview_size - 1, W // min_preview_size - 1
        steps = min(W_steps, H_steps)
        c_first_linear = self.apply_colorspace_transform(
            colorspace=colorspace, clip=clip, safe_crop=safe_crop
        )[0]
        c_first_linear = c_first_linear[:, ::steps, ::steps]
        return c_first_linear

    def generate_thumbnail(
        self,
        min_preview_size=256,
        colorspace=None,
        clip=False,
        safe_crop=0,
    ) -> np.ndarray:
        c_first_linear = self.downsize(
            min_preview_size=min_preview_size,
            colorspace=colorspace,
            clip=clip,
            safe_crop=safe_crop,
        )
        return c_first_linear

    def as_rgb(
        self,
        colorspace=None,
        dims=None,
        clip=False,
        safe_crop=0,
    ) -> np.ndarray:
        c_first_linear = self.apply_colorspace_transform(
            colorspace=colorspace, dims=dims, safe_crop=safe_crop
        )
        if clip:
            c_first_linear = np.clip(c_first_linear, 0, 1)
        return c_first_linear

    def as_sparse(
        self,
        colorspace=None,
        dims=None,
        clip=False,
        safe_crop=0,
        pattern="RGGB",
        cfa_type="bayer",
    ) -> np.ndarray:
        c_first_linear = self.apply_colorspace_transform(
            colorspace=colorspace, dims=dims, safe_crop=safe_crop
        )
        sparse = sparse_representation_three_channel(
            c_first_linear, pattern=pattern, cfa_type=cfa_type
        )
        if clip:
            sparse = np.clip(sparse, 0, 1)
        return sparse

    def as_cfa(self, **kwargs) -> np.ndarray:
        sparse = self.as_sparse(**kwargs)
        return sparse.sum(axis=0, keepdims=True)

    def as_rggb(self, cfa_type="bayer", **kwargs) -> np.ndarray:
        cfa = self.as_CFA(**kwargs)
        if cfa_type == "bayer":
            rggb = pixel_unshuffle(cfa, 2)
        else:
            rggb = pixel_unshuffle(cfa, 6)
        return rggb


class RawHandlerRawpy:
    """
    Factory class to create BaseRawHandlerRawpy instances from raw image files.
    This class handles rawpy specific parsing for pixel data and core metadata,
    and uses exifread for extracting general EXIF metadata.

    Args:
        path (string): Path to raw file.
    """

    def __new__(cls, path: str, **kwargs):
        # Use rawpy for raw pixel data and core processing metadata
        rawpy_object = rawpy.imread(path)

        # Extract Core Metadata for BaseRawHandler's processing logic
        core_metadata = CoreRawMetadata(
            black_level_per_channel=rawpy_object.black_level_per_channel,
            white_level=rawpy_object.white_level,
            rgb_xyz_matrix=rawpy_object.rgb_xyz_matrix,
            raw_pattern=rawpy_object.raw_pattern,
            camera_white_balance=np.array(rawpy_object.camera_whitebalance),
            iheight=rawpy_object.sizes.iheight,
            iwidth=rawpy_object.sizes.iwidth,
        )

        # Extract Full (General) Metadata using exifread
        metadata = get_exif_data(path)

        return BaseRawHandlerRawpy(
            rawpy_object=rawpy_object,
            core_metadata=core_metadata,
            full_metadata=metadata,
            **kwargs,
        )

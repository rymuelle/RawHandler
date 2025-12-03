
# RawHandler

[![PyPI version](https://img.shields.io/pypi/v/RawHandler.svg)](https://pypi.org/project/RawHandler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/pypi/pyversions/RawHandler.svg)](https://pypi.org/project/RawHandler/)

**RawHandler** is a lightweight wrapper around [rawpy](https://github.com/letmaik/rawpy) that provides convenient tools for working with raw sensor data, particularly for training neural networks on raw images.

---

## Features

RawHandler can:

1. **Open and convert** most camera raw files into numpy arrays.
2. **Apply black and white point correction** automatically.
3. **Provide multiple representations** of the underlying sensor data:

   * Mono Bayer representation
   * 3-channel sparse representation
   * 4-channel RGGB representation
4. **Demosaic Bayer data** using [colour-demosaicing](https://pypi.org/project/colour-demosaicing/), supporting:

   * Bilinear interpolation
   * Malvar–He–Cutler (2004)
   * DDFAPD – Menon et al. (2007)
5. **Convert color spaces** from the camera’s native space to standard targets such as XYZ, sRGB, AdobeRGB, or linear Rec.2020 — all available for every representation.
6. **Crop, resize, and generate thumbnails** while preserving Bayer pattern alignment.
7. **Read EXIF/metadata information** (ISO, shutter speed, orientation, etc.) and return it as a convenient Python dictionary.

**Currently supported:** Bayer raw images
**In progress:** Fujifilm X-Trans support

---

## Installation

You can install RawHandler directly from PyPI:

```bash
pip install RawHandler
```

Or install locally from source:

```bash
# Clone the repository
git clone https://github.com/rymuelle/RawHandler.git
cd RawHandler

# Editable/development install
pip install -e .

# Standard local install
pip install .
```

---

## Example

A simple demo notebook is available:

```text
examples/simple_demosaicing.ipynb
```

This example downloads a raw image and demonstrates the basic functionality of RawHandler.

---

## License

This project is released under the **MIT License**.

---

## Acknowledgments

Special thanks to the authors of **RawNIND**:

> Brummer, Benoit; De Vleeschouwer, Christophe, 2025.
> *Raw Natural Image Noise Dataset.*
> [https://doi.org/10.14428/DVN/DEQCIM](https://doi.org/10.14428/DVN/DEQCIM), Open Data @ UCLouvain, V1.
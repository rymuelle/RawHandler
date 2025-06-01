from RawHandler.RawHandler import RawHandler
from RawHandler.utils import download_file_requests
import os


def test_load_and_Demosaic():
    # Download example raw file.
    file_url = "https://dataverse.uclouvain.be/api/access/datafile/:persistentId?persistentId=doi:10.14428/DVN/DEQCIM/LWSLMG"
    output_file = "example_raw.arw"
    if not os.path.exists(output_file):
        download_file_requests(file_url, output_file)

    rh = RawHandler("example_raw.arw")
    rh.as_RGB_colorspace(colorspace="AdobeRGB")

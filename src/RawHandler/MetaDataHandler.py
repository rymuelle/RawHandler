import exiv2


class MetaDataHandler:
    def __init__(self, path: str):
        self.metadata = exiv2.ImageFactory.open(path)
        self.metadata.readMetadata()

    def get_ISO(self):
        return get_ISO(self.metadata)


def get_ISO(metadata):
    search_tags = [
        "Exif.Photo.RecommendedExposureIndex",
        "Exif.Photo.StandardOutputSensitivity",
        "Exif.NikonIi.ISO",
        "Exif.Nikon3.ISOSpeed",
        "Exif.Nikon3.ISOSettings",
        "Exif.Photo.ISOSpeedRatings",
        "Exif.Image.ISOSpeedRatings",
    ]

    exif = metadata.exifData()

    for tag in search_tags:
        if tag in exif:
            try:
                val = exif[tag].print()
                if val and val != 65535:
                    return float(val)
            except Exception as e:
                print(e)
                val = -1
    return val

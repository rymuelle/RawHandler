
import tifffile
import numpy as np
def convert_color_matrix(matrix):
  """
  Converts a 3x3 NumPy matrix of floats into a list of integer pairs.

  Each float value in the matrix is converted to a fractional representation
  with a denominator of 10000. The numerator is calculated by scaling the
  float value by 10000 and rounding to the nearest integer.

  Args:
    matrix: A 3x3 NumPy array with floating-point numbers.

  Returns:
    A list of 9 lists, where each inner list contains two integers
    representing the numerator and denominator.
  """
  # Ensure the input is a NumPy array
  if not isinstance(matrix, np.ndarray):
    raise TypeError("Input must be a NumPy array.")

  # Flatten the 3x3 matrix into a 1D array of 9 elements
  flattened_matrix = matrix.flatten()

  # Initialize the list for the converted matrix
  converted_list = []
  denominator = 10000

  # Iterate over each element in the flattened matrix
  for element in flattened_matrix:
    # Scale the element, round it to the nearest integer, and cast to int
    numerator = int(round(element * denominator))
    # Append the [numerator, denominator] pair to the result list
    converted_list.append([numerator, denominator])

  return converted_list

def get_ratios(string, rh):
    return [x.as_integer_ratio() for x in rh.full_metadata[string].values]


def get_as_shot_neutral(rh, denominator=10000):

    cam_mul = rh.core_metadata.camera_white_balance
    
    if cam_mul[0] == 0 or cam_mul[2] == 0:
        return [[denominator, denominator], [denominator, denominator], [denominator, denominator]]

    r_neutral = cam_mul[1] / cam_mul[0]
    g_neutral = 1.0 
    b_neutral = cam_mul[1] / cam_mul[2]

    return [
        [int(round(r_neutral * denominator)), denominator],
        [int(round(g_neutral * denominator)), denominator],
        [int(round(b_neutral * denominator)), denominator],
    ]
def convert_ccm_to_rational(matrix_3x3, denominator=10000):

    numerator_matrix = np.round(matrix_3x3 * denominator).astype(int)
    numerators_flat = numerator_matrix.flatten()
    ccm_rational = [[num, denominator] for num in numerators_flat]
    
    return ccm_rational

def get_bits_and_sampleformat(img, samples_per_pixel):
    """
    Determine BitsPerSample and SampleFormat from NumPy dtype.
    Returns (bits_per_sample, sample_format)
    """

    dtype = img.dtype

    if np.issubdtype(dtype, np.unsignedinteger):
        sample_format = 1  # Unsigned
        bits = dtype.itemsize * 8

    elif np.issubdtype(dtype, np.signedinteger):
        sample_format = 2  # Signed
        bits = dtype.itemsize * 8

    elif np.issubdtype(dtype, np.floating):
        sample_format = 3  # IEEE float
        bits = dtype.itemsize * 8

    else:
        raise ValueError(f"Unsupported image dtype: {dtype}")

    # BitsPerSample must be a list when SamplesPerPixel > 1
    if samples_per_pixel > 1:
        bits_per_sample = [bits] * samples_per_pixel
    else:
        bits_per_sample = bits

    return bits_per_sample, sample_format


def exiv2_value_to_tiff(val):
    """
    Convert exiv2.Value to a Python value suitable for tifffile extratags.
    """
    if val.count() == 1:
        return val.toLong() if val.typeName() in ('Short', 'Long') else val.toFloat()
    return [v for v in val]

def make_xmp_iso(iso):
    return f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description
    xmlns:exifEX="http://cipa.jp/exif/1.0/"
    exifEX:StandardOutputSensitivity="{iso}"/>
 </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

def copy_exif_to_dng(rh):
    """
    Convert selected exiv2 ExifData entries to tifffile extratags.
    Returns a list suitable for tifffile.imwrite(extratags=...).
    """
    exifData = rh.full_metadata.metadata.exifData()
    tags = []

    def ascii_tag(tag, exif_key):
        if exif_key in exifData:
            val = str(exifData[exif_key].value())
            tags.append((tag, 2, len(val) + 1, val, True))

    def short_tag(tag, exif_key):
        if exif_key in exifData:
            val = int(exifData[exif_key].toInt64())
            tags.append((tag, 3, 1, val, True))

    def long_tag(tag, exif_key):
        if exif_key in exifData:
            val = int(exifData[exif_key].toInt64())
            tags.append((tag, 4, 1, val, True))

    def rational_tag(tag, exif_key, scale=1):
        if exif_key in exifData:
            v = exifData[exif_key].value()
            num, den = v.toRational()
            tags.append((tag, 5, 1, [num, den], True))

    # --- ASCII ---
    ascii_tag(271, 'Exif.Image.Make')
    ascii_tag(272, 'Exif.Image.Model')
    ascii_tag(306, 'Exif.Image.DateTime')
    ascii_tag(315, 'Exif.Image.Artist')
    ascii_tag(33432, 'Exif.Image.Copyright')
    ascii_tag(42036, 'Exif.Photo.LensModel')
    ascii_tag(42035, 'Exif.Photo.LensMake')
    ascii_tag(42034, 'Exif.Photo.LensSerialNumber')

    # --- Numerical ---
    short_tag(274, 'Exif.Image.Orientation')
    iso_val = int(rh.full_metadata.get_ISO())
    tags.append((34855, 4, 1, iso_val, True))
    tags.append((34864, 3, 1, 3, True)) # SensitivityType
    tags.append((34865, 4, 1, iso_val, True)) # StandardOutputSensitivity
    # --- Rationals ---
    rational_tag(33434, 'Exif.Photo.ExposureTime')
    rational_tag(33437, 'Exif.Photo.FNumber')
    rational_tag(37386, 'Exif.Photo.FocalLength')
    rational_tag(50730, 'Exif.Photo.ExposureBiasValue')

    # tags.append((50730, 10, 1, [0, 100], True))

    # Exif.Photo.ExposureBiasValue
    return tags

def calculate_forward_matrix(color_matrix_3x3, as_shot_neutral):
    """
    Calculates ForwardMatrix1 from ColorMatrix1 and AsShotNeutral.
    color_matrix_3x3: The 3x3 matrix from rh.core_metadata.rgb_xyz_matrix
    as_shot_neutral: List of 3 floats [r, g, b] from your white balance logic
    """
    # 1. Invert the Color Matrix (Camera -> XYZ)
    # Note: If your matrix is already XYZ -> Cam, use it. 
    # If it's Cam -> XYZ, don't invert. 
    # rawpy's rgb_xyz_matrix is usually XYZ -> Cam
    try:
        cam_to_xyz = np.linalg.inv(color_matrix_3x3)
    except np.linalg.LinAlgError:
        return None

    # 2. Create the White Balance Diagonal Matrix (D)
    # ForwardMatrix expects white-balanced coordinates.
    wb_matrix = np.diag(as_shot_neutral)

    # 3. Chromatic Adaptation Matrix (Bradford transform from D65 to D50)
    # This is a standard constant used in the DNG spec
    d65_to_d50_bradford = np.array([
        [ 1.0125,  0.0333, -0.0187],
        [ 0.0104,  0.9943, -0.0053],
        [ 0.0000, -0.0012,  1.0561]
    ])

    # 4. Calculate Forward Matrix: CAT * CamToXYZ * WB
    forward_matrix = d65_to_d50_bradford @ cam_to_xyz @ wb_matrix
    
    return forward_matrix

def to_dng(rh, filepath, uint_img=None):
    """
    Saves image as a DNG-compatible TIFF.
        rh: RawHandlerRawpy Object
        filepath: location to save the output
        uint_img: (optional) Data to save, should either be a mono CFA image of the same CFA as the original image, or a three channel image
    """
    if uint_img is None:
        uint_img = rh.rawpy_object.raw_image
    uint_img = uint_img.astype(np.uint16)

    is_CFA = True if len(uint_img.shape) == 2 else False
    if is_CFA:
        samples_per_pixel = 1
        # 32803 is the code for Color Filter Array
        photometric = 32803 
    else:
        samples_per_pixel = 3
        # 34892 for Linear Raw (or 1 for BlackIsZero)
        photometric = 34892 
    #   t.set(Tag.BitsPerSample, [bpp, bpp, bpp]) # 3 channels for RGB
    #   t.set(Tag.SamplesPerPixel, 3) # 3 for RGB
    #   t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Linear_Raw)
    #   t.set(Tag.BlackLevel,[0,0,0])
    #   t.set(Tag.WhiteLevel, [65535, 65535, 65535])
    tags = []
    # Basic Geometry
    height, width = uint_img.shape[:2]
    
    # Black and White Levels
    if is_CFA:
        bl = rh.core_metadata.black_level_per_channel
        wl = rh.core_metadata.white_level
    else:
        # Do I need more control from 3 channel images? 
        bl = rh.core_metadata.black_level_per_channel
        wl = rh.core_metadata.white_level

    # DNG Metadata Mapping
    # Format: (code, data_type, count, value, writeonce)
    # Types: 3=short, 4=long, 5=rational, 2=ascii, 12=double
    
    # DNG Version (1.4.0.0) - Type 1 (BYTE)
    tags.append((50706, 1, 4, [1, 4, 0, 0], True)) 

    # ColorMatrix1 (Tag 50721)
    # Type 10 = SRATIONAL. Count = 9 elements (each is a [num, den] pair)
    ccm1 = convert_color_matrix(rh.core_metadata.rgb_xyz_matrix[:3])
    ccm1_flat = np.array(ccm1).flatten().tolist()
    tags.append((50721, 10, 9, np.array(ccm1_flat, dtype=np.int32), True))

    # AsShotNeutral (Tag 50728)
    # Type 5 = RATIONAL. Count = 3 elements
    wb = get_as_shot_neutral(rh)
    wb_flat = np.array(wb).flatten().tolist()
    tags.append((50728, 5, 3, wb_flat, True))

    # BlackLevel (Tag 50714)
    # RATIONAL (5), count is 4 for Bayer
    if any(isinstance(i, list) for i in bl) or (isinstance(bl, np.ndarray) and bl.ndim > 1):
        bl_flat = np.array(bl).flatten().tolist()
    else:
        bl_flat = []
        for val in bl:
            bl_flat.extend([int(val), 1])
    if is_CFA:
        tags.append((50713, 3, 2, [2, 2], True))
        tags.append((50714, 5, len(bl_flat)//2, bl_flat, True))
        # WhiteLevel (Tag 50717) 
        # LONG (4) or SHORT (3)
        tags.append((50717, 3, 1, int(wl), True))
    else:
        # Assuming bl was [R, G1, G2, B]
        
        if len(bl) == 4:
            bl_rgb = [bl[0], (bl[1] + bl[2]) / 2, bl[3]]
        else:
            bl_rgb = bl[:3]
                
        bl_flat = []
        for val in bl_rgb:
            bl_flat.extend([int(val), 1])
        tags.append((50714, 5, 3, bl_flat, True)) # Tag 50714: BlackLevel
        tags.append((50717, 5, 3, [int(wl), 1, int(wl), 1, int(wl), 1], True)) # WhiteLevel

    if is_CFA:
        # CFARepeatPatternDim: [Rows, Cols] -> Type 3 (SHORT)
        dims = rh.core_metadata.raw_pattern.shape
        tags.append((33421, 3, 2, dims, True))
        
        # CFAPattern: [0, 1, 1, 2] for RGGB -> Type 1 (BYTE)
        # 0=Red, 1=Green, 2=Blue
        pattern = rh.core_metadata.raw_pattern.flatten()
        pattern[pattern==3] = 1
        tags.append((33422, 1, len(pattern), pattern, True))

        #CFAPlaneColor
        tags.append((50710, 1, 3, [0, 1, 2], True))
        #CFALayout
        tags.append((50711, 3, 1, 1, True))

    
    #D65 Illuminant
    tags.append((50723, 3, 1, 21, True))

    exif_tags = copy_exif_to_dng(rh)
    # 4. Save the file
    planarconfig = None if is_CFA else 1
    tifffile.imwrite(
        filepath,
        uint_img,
        photometric=photometric,
        planarconfig=planarconfig,
        extrasamples=[],
        extratags=tags + exif_tags
    )
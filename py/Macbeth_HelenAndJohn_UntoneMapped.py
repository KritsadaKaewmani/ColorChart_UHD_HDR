try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False
    print("Warning: OpenEXR or Imath module not found. EXR output will be skipped.")

try:
    import numpy as np
except ImportError:
    print("\nCRITICAL ERROR: 'numpy' module not found.")
    print("Please run this script using the provided helper script:")
    print("  ./ColorChart/run_with_venv.sh")
    print("Or use the virtual environment python directly:")
    print("  ../.venv/bin/python ColorChart/Macbeth_HelenAndJohn_UntoneMapped.py")
    exit(1)

from PIL import Image, ImageDraw, ImageFont

# --- User parameters ---
# exr_path = '/Users/kkaewmani/Desktop/Data/Python_Works/ColorChart/macbeth_HelenAndJohn_ACES20_2065-1_ref.exr'  # Path to your EXR file
chart_rows = 4
chart_cols = 6

#RGB values (half float, ACES AP0, linear): 
#[[0.17017, 0.12756, 0.09253], [0.51416, 0.39014, 0.30615], [0.20227, 0.21912, 0.38501], [0.12732, 0.16101, 0.07190], [0.27344, 0.25024, 0.45044], [0.33911, 0.50146, 0.46289]],
#[[0.54102, 0.30664, 0.08466], [0.17114, 0.15796, 0.48120], [0.38013, 0.16675, 0.16333], [0.09503, 0.06580, 0.14160], [0.39136, 0.50488, 0.12177], [0.51611, 0.38110, 0.08325]],
#[[0.09924, 0.07831, 0.36035], [0.17712, 0.32642, 0.10522], [0.26440, 0.09515, 0.07050], [0.67920, 0.62842, 0.10675], [0.34302, 0.15735, 0.31982], [0.14844, 0.23254, 0.40088]],
#[[1.14258, 1.13867, 1.19141], [0.68408, 0.68359, 0.72412], [0.40039, 0.39990, 0.42358], [0.19910, 0.19800, 0.20972], [0.09064, 0.09100, 0.09698], [0.03406, 0.03363, 0.03680]],
#[[1.14258, 1.13867, 1.19141], [0.68408, 0.68359, 0.72412], [0.40039, 0.39990, 0.42358], [0.19910, 0.19800, 0.20972], [0.09064, 0.09100, 0.09698], [0.03406, 0.03363, 0.03680]],

# reference values (physical properties) 
# Patch Number  Name            Reflectance Factor (Y in CIE xyY)   Approximate Luminance (%)
# 19		    White		    0.900		                        90.0%
# 20		    Neutral 8	    0.591		                        59.1%
# 21		    Neutral 6.5	    0.362		                        36.2%
# 22		    Neutral 5	    0.198		                        19.8%
# 23		    Neutral 3.5	    0.090		                        9.0%
# 24		    Black		    0.031		                        3.1%
# Borrow Ham Natdanai's light meter and OneCool spectrometer to confirm the values
# Cine Meter App 102,000 lux (30,900 / 1,310 nits)

def create_text_mask(width, height, text, font_size=40, margin=10):
    # Create a blank image for the text mask
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except IOError:
        # Fallback to default if specific font not found
        font = ImageFont.load_default()
        print("Warning: Menlo font not found, using default.")

    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (width - text_w) // 2
    y = height - text_h - margin
    
    # Draw text
    draw.text((x, y), text, font=font, fill=255)
    
    return np.array(img) > 128
# --- Hardcoded RGB values (half float, ACES AP0, linear) ---
rgb_array = np.array([
    [[0.17017, 0.12756, 0.09253], [0.51416, 0.39014, 0.30615], [0.20227, 0.21912, 0.38501], [0.12732, 0.16101, 0.07190], [0.27344, 0.25024, 0.45044], [0.33911, 0.50146, 0.46289]],
    [[0.54102, 0.30664, 0.08466], [0.17114, 0.15796, 0.48120], [0.38013, 0.16675, 0.16333], [0.09503, 0.06580, 0.14160], [0.39136, 0.50488, 0.12177], [0.51611, 0.38110, 0.08325]],
    [[0.09924, 0.07831, 0.36035], [0.17712, 0.32642, 0.10522], [0.26440, 0.09515, 0.07050], [0.67920, 0.62842, 0.10675], [0.34302, 0.15735, 0.31982], [0.14844, 0.23254, 0.40088]],
    [[1.14258, 1.13867, 1.19141], [0.68408, 0.68359, 0.72412], [0.40039, 0.39990, 0.42358], [0.19910, 0.19800, 0.20972], [0.09064, 0.09100, 0.09698], [0.03406, 0.03363, 0.03680]]
], dtype=np.float16)

# --- Print as 4x6 array ---
print("RGB values (half float, ACES AP0, linear):")
for row in range(chart_rows):
    print("[", end="")
    for col in range(chart_cols):
        rgb = rgb_array[row, col]
        print(f"[{rgb[0]:.5f}, {rgb[1]:.5f}, {rgb[2]:.5f}]", end="")
        if col < chart_cols - 1:
            print(", ", end="")
    print("],")

# --- Generate New EXR (3840x2560) ---
if OPENEXR_AVAILABLE:
    print("\nGenerating macbeth_HelenAndJohn_UntoneMapped_ACES20_2065-1.exr...")

    # Target dimensions
    target_width = 3840
    target_height = 2560

    # Layout parameters (same as macbeth_colorchecker.py)
    rows, cols = 4, 6
    spacing = 50
    patch_width = (target_width - (cols + 1) * spacing) // cols
    patch_height = (target_height - (rows + 1) * spacing) // rows

    # Create new image buffer
    new_exr_r = np.full((target_height, target_width), 0.02, dtype=np.float16)
    new_exr_g = np.full((target_height, target_width), 0.02, dtype=np.float16)
    new_exr_b = np.full((target_height, target_width), 0.02, dtype=np.float16)

    # Fill patches
    for i in range(rows):
        for j in range(cols):
            # Get color from extracted array
            color = rgb_array[i, j]
            
            x0 = spacing + j * (patch_width + spacing)
            y0 = spacing + i * (patch_height + spacing)
            x1 = x0 + patch_width
            y1 = y0 + patch_height
            
            new_exr_r[y0:y1, x0:x1] = color[0]
            new_exr_g[y0:y1, x0:x1] = color[1]
            new_exr_b[y0:y1, x0:x1] = color[2]

    # Add filename text to EXR
    output_filename = "macbeth_HelenAndJohn_UntoneMapped_ACES20_2065-1.exr"
    text_mask = create_text_mask(target_width, target_height, output_filename)
    text_color_exr = 0.6

    new_exr_r[text_mask] = text_color_exr
    new_exr_g[text_mask] = text_color_exr
    new_exr_b[text_mask] = text_color_exr

    # Write EXR
    header = OpenEXR.Header(target_width, target_height)
    header['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}

    # Add ACES 2065-1 (AP0) chromaticities
    # ACES AP0 primaries and D60 white point
    header['chromaticities'] = Imath.Chromaticities(
        Imath.V2f(0.7347, 0.2653),  # Red primary
        Imath.V2f(0.0000, 1.0000),  # Green primary
        Imath.V2f(0.0001, -0.0770), # Blue primary
        Imath.V2f(0.32168, 0.33767) # D60 white point
    )

    # ACES Image Container Flag (SMPTE ST 2065-4)
    header['acesImageContainerFlag'] = 1

    out_exr = OpenEXR.OutputFile(output_filename, header)
    out_exr.writePixels({'R': new_exr_r.tobytes(), 'G': new_exr_g.tobytes(), 'B': new_exr_b.tobytes()})
    out_exr.close()

    print(f"Image saved as {output_filename} with ACES 2065-1 chromaticities")
else:
    print("\nSkipping EXR generation (OpenEXR module not available).")

# --- Generate Rec.2020 PQ TIFF (16-bit) ---
print("\nGenerating macbeth_HelenAndJohn_UntoneMapped_Rec2020_PQ.tiff...")

from PIL import Image
import subprocess
import os

# ICC Profile Path
ICC_PROFILE_REC2020_PQ = "/Library/Application Support/Adobe/Color/Profiles/HDR_UHD_ST2084.icc"

# Load ICC profile
try:
    with open(ICC_PROFILE_REC2020_PQ, 'rb') as f:
        REC2020_PQ_PROFILE = f.read()
    print(f"✓ Loaded Rec.2020 PQ profile: {ICC_PROFILE_REC2020_PQ}")
except Exception as e:
    print(f"⚠ Failed to load Rec.2020 PQ profile: {e}")
    REC2020_PQ_PROFILE = None

# PQ Constants (SMPTE ST 2084)
m1 = 0.1593017578125
m2 = 78.84375
c1 = 0.8359375
c2 = 18.8515625
c3 = 18.6875

def luminance_to_pq(lum):
    # Normalize luminance to [0,1] for PQ formula (10,000 nits reference)
    # Avoid negative values or zero division issues if any
    lum = max(0.0, lum)
    L = lum / 10000.0
    numerator = c1 + c2 * (L ** m1)
    denominator = 1 + c3 * (L ** m1)
    pq = (numerator / denominator) ** m2
    return pq

# ACES AP0 to Rec.2020 Matrix (Linear, Bradford Adapted)
M_AP0_to_2020 = np.array([
    [ 1.49040952, -0.26617092, -0.2242386 ],
    [-0.0801675,  1.18216712, -0.10199962],
    [ 0.00322763, -0.03477648,  1.03154884]
])

def apply_matrix(rgb, matrix):
    return matrix.dot(rgb)

# Calculate background color for TIFF (0.02 ACES AP0)
bg_aces = np.array([0.02, 0.02, 0.02])
bg_rec2020 = M_AP0_to_2020.dot(bg_aces)
bg_rec2020 = np.maximum(0, bg_rec2020)
bg_lum = bg_rec2020 * 203.0
bg_pq = np.array([luminance_to_pq(c) for c in bg_lum])
bg_val = (bg_pq * 65535).astype(np.uint16)

print(f"Background values (16-bit): R={bg_val[0]}, G={bg_val[1]}, B={bg_val[2]}")

# Create TIFF buffer (uint16)
tiff_r = np.full((target_height, target_width), bg_val[0], dtype=np.uint16)
tiff_g = np.full((target_height, target_width), bg_val[1], dtype=np.uint16)
tiff_b = np.full((target_height, target_width), bg_val[2], dtype=np.uint16)


# Fill patches
for i in range(rows):
    for j in range(cols):
        # Get ACES AP0 color
        aces_rgb = rgb_array[i, j]
        
        # Convert to Rec.2020 Linear
        rec2020_rgb = M_AP0_to_2020.dot(aces_rgb)
        
        # Clamp negative values
        rec2020_rgb = np.maximum(0, rec2020_rgb)
        
        # Scale to Luminance (1.0 = 203 nits)
        lum_rgb = rec2020_rgb * 203.0
        
        # Apply PQ
        pq_r = luminance_to_pq(lum_rgb[0])
        pq_g = luminance_to_pq(lum_rgb[1])
        pq_b = luminance_to_pq(lum_rgb[2])
        
        # Quantize to 16-bit
        val_r = int(round(pq_r * 65535))
        val_g = int(round(pq_g * 65535))
        val_b = int(round(pq_b * 65535))
        
        # Fill patch
        x0 = spacing + j * (patch_width + spacing)
        y0 = spacing + i * (patch_height + spacing)
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        
        tiff_r[y0:y1, x0:x1] = val_r
        tiff_g[y0:y1, x0:x1] = val_g
        tiff_b[y0:y1, x0:x1] = val_b

# Create separate copies for TIFF and PNG exports (before adding any text)
# TIFF arrays - will have TIFF filename added
tiff_export_r = tiff_r.copy()
tiff_export_g = tiff_g.copy()
tiff_export_b = tiff_b.copy()

# PNG arrays - will have PNG filename added
png_export_r = tiff_r.copy()
png_export_g = tiff_g.copy()
png_export_b = tiff_b.copy()

# Add filename text to TIFF arrays
output_tiff = "macbeth_HelenAndJohn_UntoneMapped_Rec2020_PQ_W203.tiff"
text_mask_tiff = create_text_mask(target_width, target_height, output_tiff)
text_color_tiff = int(0.6 * 65535)

tiff_export_r[text_mask_tiff] = text_color_tiff
tiff_export_g[text_mask_tiff] = text_color_tiff
tiff_export_b[text_mask_tiff] = text_color_tiff

# Save using PIL split + ImageMagick (same as macbeth_colorchecker.py)
r_filename = "temp_R.tiff"
g_filename = "temp_G.tiff"
b_filename = "temp_B.tiff"

# Save TIFF channels with TIFF filename text
Image.fromarray(tiff_export_r).convert('I;16').save(r_filename)
Image.fromarray(tiff_export_g).convert('I;16').save(g_filename)
Image.fromarray(tiff_export_b).convert('I;16').save(b_filename)

try:
    cmd = ["magick", r_filename, g_filename, b_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_tiff]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        cmd = ["convert", r_filename, g_filename, b_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_tiff]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
    if result.returncode == 0:
        print(f"Successfully created {output_tiff} with ICC profile")
        
        # --- Generate PNG with PNG filename only ---
        output_png = output_tiff.replace(".tiff", ".png")
        print(f"Generating {output_png}...")
        
        # Add PNG filename text to PNG arrays (separate from TIFF)
        text_mask_png = create_text_mask(target_width, target_height, output_png)
        
        png_export_r[text_mask_png] = text_color_tiff
        png_export_g[text_mask_png] = text_color_tiff
        png_export_b[text_mask_png] = text_color_tiff
        
        # Save temp channels for PNG
        r_png_filename = "temp_png_R.tiff"
        g_png_filename = "temp_png_G.tiff"
        b_png_filename = "temp_png_B.tiff"
        
        Image.fromarray(png_export_r).convert('I;16').save(r_png_filename)
        Image.fromarray(png_export_g).convert('I;16').save(g_png_filename)
        Image.fromarray(png_export_b).convert('I;16').save(b_png_filename)
        
        cmd_png = ["magick", r_png_filename, g_png_filename, b_png_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_png]
        result_png = subprocess.run(cmd_png, capture_output=True, text=True)
        
        if result_png.returncode != 0:
             cmd_png = ["convert", r_png_filename, g_png_filename, b_png_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_png]
             result_png = subprocess.run(cmd_png, capture_output=True, text=True)
             
        if result_png.returncode == 0:
            print(f"Successfully created {output_png} with ICC profile")
        else:
            print(f"Error creating PNG: {result_png.stderr}")

        # Cleanup
        for f in [r_filename, g_filename, b_filename, r_png_filename, g_png_filename, b_png_filename]:
            if os.path.exists(f):
                os.remove(f)
    else:
        print(f"Error combining TIFFs: {result.stderr}")

except Exception as e:
    print(f"Failed to run ImageMagick: {e}")
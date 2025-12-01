from PIL import Image, ImageDraw, ImageFont, ImageCms
import numpy as np
import base64
import io
import subprocess
import os

# Image and chart parameters
img_width, img_height = 3840, 2560
rows, cols = 4, 6
spacing = 50  # pixels between patches

# sRGB reference values for Macbeth ColorChecker (row-major order)
patch_colors = [
    (115, 82, 68),   (194, 150, 130), (98, 122, 157),  (87, 108, 67),   (133, 128, 177), (103, 189, 170),
    (214, 126, 44),  (80, 91, 166),   (193, 90, 99),   (94, 60, 108),   (157, 188, 64),  (224, 163, 46),
    (56, 61, 150),   (70, 148, 73),   (175, 54, 60),   (231, 199, 31),  (187, 86, 149),  (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85),    (52, 52, 52)
]

# ============================================================================
# ICC Profile Configuration
# ============================================================================
# Configure ICC profile paths here for easy management
# Paths point to macOS system profiles, modify as needed for your system
# /Library/ColorSync/Profiles/Displays/HDR_P3_D65_ST2084.icc //legacy for ref,use p3_PQ instead

ICC_PROFILE_SRGB = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
ICC_PROFILE_DISPLAY_P3 = "/System/Library/ColorSync/Profiles/Display P3.icc"
ICC_PROFILE_DISPLAY_P3_PQ = "/Library/Application Support/Adobe/Color/Profiles/P3_PQ.icc"  # HDR Display P3 with PQ
ICC_PROFILE_REC2020_PQ = "/Library/Application Support/Adobe/Color/Profiles/HDR_UHD_ST2084.icc"

# Load ICC profiles at startup (as bytes for embedding)
try:
    with open(ICC_PROFILE_SRGB, 'rb') as f:
        SRGB_PROFILE = f.read()
    print(f"✓ Loaded sRGB profile: {ICC_PROFILE_SRGB}")
except Exception as e:
    print(f"⚠ Failed to load sRGB profile: {e}")
    SRGB_PROFILE = None

try:
    with open(ICC_PROFILE_DISPLAY_P3, 'rb') as f:
        DISPLAY_P3_PROFILE = f.read()
    print(f"✓ Loaded Display P3 profile: {ICC_PROFILE_DISPLAY_P3}")
except Exception as e:
    print(f"⚠ Failed to load Display P3 profile: {e}")
    DISPLAY_P3_PROFILE = None

try:
    with open(ICC_PROFILE_DISPLAY_P3_PQ, 'rb') as f:
        DISPLAY_P3_PQ_PROFILE = f.read()
    print(f"✓ Loaded Display P3 PQ profile: {ICC_PROFILE_DISPLAY_P3_PQ}")
except Exception as e:
    print(f"⚠ Failed to load Display P3 PQ profile: {e}")
    DISPLAY_P3_PQ_PROFILE = None

try:
    with open(ICC_PROFILE_REC2020_PQ, 'rb') as f:
        REC2020_PQ_PROFILE = f.read()
    print(f"✓ Loaded Rec.2020 PQ profile: {ICC_PROFILE_REC2020_PQ}")
except Exception as e:
    print(f"⚠ Failed to load Rec.2020 PQ profile: {e}")
    REC2020_PQ_PROFILE = None

print()  # Blank line for readability

# ============================================================================


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

def srgb_to_linear(srgb_val):
    # sRGB to Linear sRGB
    # srgb_val is in [0, 255]
    u = srgb_val / 255.0
    if u <= 0.04045:
        return u / 12.92
    else:
        return ((u + 0.055) / 1.055) ** 2.4

def linear_srgb_to_rec2020(r, g, b):
    # Matrix from Linear sRGB to Linear Rec.2020
    # Source: http://www.brucelindbloom.com/index.html?Eqn_RGB_to_RGB.html
    # sRGB primaries to XYZ, then XYZ to Rec.2020 primaries
    
    # Linear sRGB to XYZ (D65)
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    # XYZ to Linear Rec.2020 (D65)
    r_2020 = 1.7166511880 * X - 0.3556707838 * Y - 0.2533662814 * Z
    g_2020 = -0.6666843518 * X + 1.6164812366 * Y + 0.0157685458 * Z
    b_2020 = 0.0176398574 * X - 0.0427706533 * Y + 0.9421031254 * Z
    
    return r_2020, g_2020, b_2020

def linear_srgb_to_display_p3(r, g, b):
    # Matrix from Linear sRGB to Linear Display P3
    # Both use D65 white point
    
    # Linear sRGB to XYZ (D65)
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    # XYZ to Linear Display P3 (D65)
    # Display P3 uses DCI-P3 primaries with D65 white point
    r_p3 = 2.4934969119 * X - 0.9313836179 * Y - 0.4027107845 * Z
    g_p3 = -0.8294889696 * X + 1.7626640603 * Y + 0.0236246858 * Z
    b_p3 = 0.0358458302 * X - 0.0761723893 * Y + 0.9568845240 * Z
    
    return r_p3, g_p3, b_p3

def create_srgb_profile():
    """Return the pre-loaded sRGB ICC profile"""
    return SRGB_PROFILE

def create_display_p3_profile():
    """Return the pre-loaded Display P3 ICC profile"""
    return DISPLAY_P3_PROFILE

def create_display_p3_pq_profile():
    """Return the pre-loaded Display P3 PQ HDR ICC profile"""
    return DISPLAY_P3_PQ_PROFILE

def create_rec2020_pq_profile():
    """Return the pre-loaded Rec.2020 PQ ICC profile"""
    return REC2020_PQ_PROFILE


def apply_gamma_display_p3(linear_val):
    """Apply Display P3 gamma (same as sRGB) to linear value"""
    if linear_val <= 0.0031308:
        return 12.92 * linear_val
    else:
        return 1.055 * (linear_val ** (1.0/2.4)) - 0.055

# PQ constants (SMPTE ST 2084)
m1 = 0.1593017578125
m2 = 78.84375
c1 = 0.8359375
c2 = 18.8515625
c3 = 18.6875

def luminance_to_pq(lum):
    # Normalize luminance to [0,1] for PQ formula (10,000 nits reference)
    L = lum / 10000.0
    numerator = c1 + c2 * (L ** m1)
    denominator = 1 + c3 * (L ** m1)
    pq = (numerator / denominator) ** m2
    return pq

# Reference white luminance for SDR
ref_white_nits_sdr = 100.0
# Reference white luminance for HDR
ref_white_nits_hdr = 203.0



# Calculate patch size
patch_width = (img_width - (cols + 1) * spacing) // cols
patch_height = (img_height - (rows + 1) * spacing) // rows

# Create image with neutral background
img = Image.new("RGB", (img_width, img_height), (40, 40, 40))
draw = ImageDraw.Draw(img)

# Draw patches
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        color = patch_colors[idx]
        x0 = spacing + j * (patch_width + spacing)
        y0 = spacing + i * (patch_height + spacing)
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        draw.rectangle([x0, y0, x1, y1], fill=color)

# Add filename text to sRGB PNG
output_srgb_filename = "macbeth_colorchecker_SDR_W100_sRGB.png"
text_mask_srgb = create_text_mask(img_width, img_height, output_srgb_filename)
text_color_srgb = int(0.6 * 255)

# Convert mask to RGB and apply to image
for y in range(img_height):
    for x in range(img_width):
        if text_mask_srgb[y, x]:
            img.putpixel((x, y), (text_color_srgb, text_color_srgb, text_color_srgb))

# Save the image with sRGB ICC profile
srgb_profile = create_srgb_profile()
if srgb_profile:
    img.save(output_srgb_filename, icc_profile=srgb_profile)
    print(f"Image saved as {output_srgb_filename} with sRGB ICC profile")
else:
    img.save(output_srgb_filename)
    print(f"Image saved as {output_srgb_filename} (ICC profile not embedded)")


# --- Display P3 SDR Export ---

# Create Display P3 image (8-bit for PNG, will also create 16-bit for TIFF)
img_p3_array_16bit = np.zeros((img_height, img_width, 3), dtype=np.uint16)
img_p3_array_8bit = np.zeros((img_height, img_width, 3), dtype=np.uint8)

# Fill background first
bg_lin = srgb_to_linear(40)
bg_r_p3, bg_g_p3, bg_b_p3 = linear_srgb_to_display_p3(bg_lin, bg_lin, bg_lin)
# Clamp and apply gamma
bg_r_p3 = max(0, min(1, bg_r_p3))
bg_g_p3 = max(0, min(1, bg_g_p3))
bg_b_p3 = max(0, min(1, bg_b_p3))
bg_r_gamma = apply_gamma_display_p3(bg_r_p3)
bg_g_gamma = apply_gamma_display_p3(bg_g_p3)
bg_b_gamma = apply_gamma_display_p3(bg_b_p3)

bg_val_16bit = [int(round(bg_r_gamma * 65535)), int(round(bg_g_gamma * 65535)), int(round(bg_b_gamma * 65535))]
bg_val_8bit = [int(round(bg_r_gamma * 255)), int(round(bg_g_gamma * 255)), int(round(bg_b_gamma * 255))]

img_p3_array_16bit[:, :] = bg_val_16bit
img_p3_array_8bit[:, :] = bg_val_8bit

# Draw patches in Display P3
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        srgb_color = patch_colors[idx]
        
        # 1. sRGB to Linear sRGB
        lin_r = srgb_to_linear(srgb_color[0])
        lin_g = srgb_to_linear(srgb_color[1])
        lin_b = srgb_to_linear(srgb_color[2])
        
        # 2. Linear sRGB to Linear Display P3
        p3_r, p3_g, p3_b = linear_srgb_to_display_p3(lin_r, lin_g, lin_b)
        
        # Clamp to valid range [0, 1]
        p3_r = max(0, min(1, p3_r))
        p3_g = max(0, min(1, p3_g))
        p3_b = max(0, min(1, p3_b))
        
        # 3. Apply Display P3 gamma (same as sRGB gamma)
        p3_r_gamma = apply_gamma_display_p3(p3_r)
        p3_g_gamma = apply_gamma_display_p3(p3_g)
        p3_b_gamma = apply_gamma_display_p3(p3_b)
        
        # 4. Quantize to 16-bit and 8-bit
        val_r_16 = int(round(p3_r_gamma * 65535))
        val_g_16 = int(round(p3_g_gamma * 65535))
        val_b_16 = int(round(p3_b_gamma * 65535))
        
        val_r_8 = int(round(p3_r_gamma * 255))
        val_g_8 = int(round(p3_g_gamma * 255))
        val_b_8 = int(round(p3_b_gamma * 255))
        
        # Draw on arrays
        x0 = spacing + j * (patch_width + spacing)
        y0 = spacing + i * (patch_height + spacing)
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        
        img_p3_array_16bit[y0:y1, x0:x1] = [val_r_16, val_g_16, val_b_16]
        img_p3_array_8bit[y0:y1, x0:x1] = [val_r_8, val_g_8, val_b_8]

# Create separate copies for TIFF and PNG exports
tiff_p3_array = img_p3_array_16bit.copy()
png_p3_array = img_p3_array_8bit.copy()

# Add filename text to Display P3 TIFF
output_p3_tiff = "macbeth_colorchecker_SDR_W100_DisplayP3.tif"
text_mask_p3_tiff = create_text_mask(img_width, img_height, output_p3_tiff)
text_color_p3_16bit = int(0.6 * 65535)

for y in range(img_height):
    for x in range(img_width):
        if text_mask_p3_tiff[y, x]:
            tiff_p3_array[y, x] = [text_color_p3_16bit, text_color_p3_16bit, text_color_p3_16bit]

# Save Display P3 TIFF (16-bit) using ImageMagick
r_p3_filename = "temp_p3_R.tiff"
g_p3_filename = "temp_p3_G.tiff"
b_p3_filename = "temp_p3_B.tiff"

# 2. Convert to the specific 'I;16' internal format
r_ch_p3 = (Image.fromarray(tiff_p3_array[:,:,0])).convert('I;16')
g_ch_p3 = (Image.fromarray(tiff_p3_array[:,:,1])).convert('I;16')
b_ch_p3 = (Image.fromarray(tiff_p3_array[:,:,2])).convert('I;16')

r_ch_p3.save(r_p3_filename)
g_ch_p3.save(g_p3_filename)
b_ch_p3.save(b_p3_filename)

try:
    cmd = ["magick", r_p3_filename, g_p3_filename, b_p3_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_DISPLAY_P3, output_p3_tiff]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        cmd = ["convert", r_p3_filename, g_p3_filename, b_p3_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_DISPLAY_P3, output_p3_tiff]
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully created {output_p3_tiff}")
        os.remove(r_p3_filename)
        os.remove(g_p3_filename)
        os.remove(b_p3_filename)
    else:
        print(f"Error combining Display P3 TIFFs: {result.stderr}")
except Exception as e:
    print(f"Failed to run ImageMagick for Display P3 TIFF: {e}")

# Add filename text to Display P3 PNG
output_p3_png = "macbeth_colorchecker_SDR_W100_DisplayP3.png"
text_mask_p3_png = create_text_mask(img_width, img_height, output_p3_png)
text_color_p3_8bit = int(0.6 * 255)

for y in range(img_height):
    for x in range(img_width):
        if text_mask_p3_png[y, x]:
            png_p3_array[y, x] = [text_color_p3_8bit, text_color_p3_8bit, text_color_p3_8bit]

# Save Display P3 PNG (8-bit)
img_p3_png = Image.fromarray(png_p3_array, mode='RGB')
p3_profile = create_display_p3_profile()
if p3_profile:
    img_p3_png.save(output_p3_png, icc_profile=p3_profile)
    print(f"Image saved as {output_p3_png} with Display P3 ICC profile")
else:
    img_p3_png.save(output_p3_png)
    print(f"Image saved as {output_p3_png} (ICC profile not embedded)")




# --- HDR Export (Rec.2020 PQ) ---

# Create blank 16-bit image for HDR
img_hdr_array = np.zeros((img_height, img_width, 3), dtype=np.uint16)


# Process patches for HDR
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        srgb_color = patch_colors[idx]
        
        # 1. sRGB to Linear sRGB
        lin_r = srgb_to_linear(srgb_color[0])
        lin_g = srgb_to_linear(srgb_color[1])
        lin_b = srgb_to_linear(srgb_color[2])
        
        # 2. Linear sRGB to Linear Rec.2020
        rec2020_r, rec2020_g, rec2020_b = linear_srgb_to_rec2020(lin_r, lin_g, lin_b)
        
        # Clamp negative values (gamut mapping simple approach)
        rec2020_r = max(0, rec2020_r)
        rec2020_g = max(0, rec2020_g)
        rec2020_b = max(0, rec2020_b)
        
        # 3. Scale to Luminance (Reference White 203 nits)
        # Assuming (1,1,1) in Linear sRGB maps to 203 nits
        lum_r = rec2020_r * ref_white_nits_hdr
        lum_g = rec2020_g * ref_white_nits_hdr
        lum_b = rec2020_b * ref_white_nits_hdr
        
        # 4. Apply PQ OETF
        pq_r = luminance_to_pq(lum_r)
        pq_g = luminance_to_pq(lum_g)
        pq_b = luminance_to_pq(lum_b)
        
        # 5. Quantize to 16-bit
        val_r = int(round(pq_r * 65535))
        val_g = int(round(pq_g * 65535))
        val_b = int(round(pq_b * 65535))
        
        # Draw on array
        x0 = spacing + j * (patch_width + spacing)
        y0 = spacing + i * (patch_height + spacing)
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        
        img_hdr_array[y0:y1, x0:x1] = [val_r, val_g, val_b]

# Fill background (dark grey in sRGB -> HDR?)
# Original background was (40, 40, 40) sRGB
bg_lin = srgb_to_linear(40)
bg_r_2020, bg_g_2020, bg_b_2020 = linear_srgb_to_rec2020(bg_lin, bg_lin, bg_lin)
bg_lum_r = max(0, bg_r_2020) * ref_white_nits_hdr
bg_lum_g = max(0, bg_g_2020) * ref_white_nits_hdr
bg_lum_b = max(0, bg_b_2020) * ref_white_nits_hdr
bg_pq_r = int(round(luminance_to_pq(bg_lum_r) * 65535))
bg_pq_g = int(round(luminance_to_pq(bg_lum_g) * 65535))
bg_pq_b = int(round(luminance_to_pq(bg_lum_b) * 65535))

# Apply background color to empty areas (where array is still 0)
# This is a bit inefficient, better to initialize with background
# But since we already filled patches, let's just use a mask or simpler approach
# Re-initializing for cleaner code
img_hdr_array_final = np.full((img_height, img_width, 3), [bg_pq_r, bg_pq_g, bg_pq_b], dtype=np.uint16)

for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        srgb_color = patch_colors[idx]
        
        lin_r = srgb_to_linear(srgb_color[0])
        lin_g = srgb_to_linear(srgb_color[1])
        lin_b = srgb_to_linear(srgb_color[2])
        
        rec2020_r, rec2020_g, rec2020_b = linear_srgb_to_rec2020(lin_r, lin_g, lin_b)
        
        rec2020_r = max(0, rec2020_r)
        rec2020_g = max(0, rec2020_g)
        rec2020_b = max(0, rec2020_b)
        
        lum_r = rec2020_r * ref_white_nits_hdr
        lum_g = rec2020_g * ref_white_nits_hdr
        lum_b = rec2020_b * ref_white_nits_hdr
        
        pq_r = luminance_to_pq(lum_r)
        pq_g = luminance_to_pq(lum_g)
        pq_b = luminance_to_pq(lum_b)
        
        val_r = int(round(pq_r * 65535))
        val_g = int(round(pq_g * 65535))
        val_b = int(round(pq_b * 65535))
        
        x0 = spacing + j * (patch_width + spacing)
        y0 = spacing + i * (patch_height + spacing)
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        
        img_hdr_array_final[y0:y1, x0:x1] = [val_r, val_g, val_b]

# Create separate copies for TIFF and PNG preview exports (before adding any text)
# TIFF array - will have TIFF filename added
tiff_export_array = img_hdr_array_final.copy()

# PNG preview array - will have PNG filename added
png_preview_array = img_hdr_array_final.copy()

# Add filename text to TIFF array
output_tiff = "macbeth_colorchecker_HDR_W203_Rec2020_PQ.tiff"
text_mask_hdr = create_text_mask(img_width, img_height, output_tiff)
text_color_hdr = int(0.6 * 65535)

for y in range(img_height):
    for x in range(img_width):
        if text_mask_hdr[y, x]:
            tiff_export_array[y, x] = [text_color_hdr, text_color_hdr, text_color_hdr]

# Save HDR Image
# PIL mode 'I;16' is for single channel. For RGB 16-bit, we might need 'RGB' with 16-bit data?
# PIL doesn't fully support 16-bit RGB PNG writing easily from 'RGB' mode directly if not handled carefully.
# However, saving numpy array of uint16 as PNG usually works with cv2 or imageio.
# With PIL, we can try 'RGB' mode but it expects 8-bit.
# 'I;16' is 16-bit grayscale.
# For 16-bit RGB PNG, we can use:
# img = Image.fromarray(img_hdr_array_final, mode='RGB') -> this will be 8-bit if not careful.
# Actually PIL 16-bit RGB support is limited.
# Let's check if we can save it.
# Alternative: Use PyPNG or similar if PIL fails, but I don't have new libs.
# Let's try Image.fromarray with 'I;16' for each channel and merge? No.
# Recent Pillow versions support 16-bit PNGs better.
# Let's try:
# PIL does not support 16-bit RGB creation/export easily.
# Saving as 3 separate 16-bit TIFF files (R, G, B) to preserve data.
r_filename = "macbeth_colorchecker_HDR_Rec2020_PQ_R.tiff"
g_filename = "macbeth_colorchecker_HDR_Rec2020_PQ_G.tiff"
b_filename = "macbeth_colorchecker_HDR_Rec2020_PQ_B.tiff"

# Save TIFF with TIFF filename text
r_ch = (Image.fromarray(tiff_export_array[:,:,0])).convert('I;16')
g_ch = (Image.fromarray(tiff_export_array[:,:,1])).convert('I;16')
b_ch = (Image.fromarray(tiff_export_array[:,:,2])).convert('I;16')

r_ch.save(r_filename)
g_ch.save(g_filename)
b_ch.save(b_filename)
print("Saved temporary channel TIFFs.")

# Use ImageMagick to combine them into a single 16-bit RGB TIFF
import subprocess
import os

try:
    # Command: magick convert R.tiff G.tiff B.tiff -combine -depth 16 output.tiff
    # Note: 'magick' might be needed on some systems, or just 'convert'.
    # Checking if 'magick' exists or 'convert'.
    # We will try 'magick' first as it is newer, then 'convert'.
    
    cmd = ["magick", r_filename, g_filename, b_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_tiff]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # Try 'convert' if 'magick' fails (e.g. older IM)
        cmd = ["convert", r_filename, g_filename, b_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_tiff]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
    if result.returncode == 0:
        print(f"Successfully created {output_tiff}")
        # Clean up temporary files
        os.remove(r_filename)
        os.remove(g_filename)
        os.remove(b_filename)
        print("Removed temporary channel files.")
    else:
        print(f"Error combining TIFFs: {result.stderr}")

except Exception as e:
    print(f"Failed to run ImageMagick: {e}")

# Generate 8-bit PNG preview from PNG preview array (separate from TIFF)
output_preview_filename = "macbeth_colorchecker_HDR_W203_Rec2020_PQ.png"

# Add filename text to PNG preview array
text_mask_preview = create_text_mask(img_width, img_height, output_preview_filename)
text_color_preview_16bit = int(0.6 * 65535)

for y in range(img_height):
    for x in range(img_width):
        if text_mask_preview[y, x]:
            png_preview_array[y, x] = [text_color_preview_16bit, text_color_preview_16bit, text_color_preview_16bit]

# Convert to 8-bit for PNG preview
img_hdr_8bit = (png_preview_array / 256).astype(np.uint8)
img_preview = Image.fromarray(img_hdr_8bit, mode='RGB')

rec2020_pq_profile = create_rec2020_pq_profile()
if rec2020_pq_profile:
    img_preview.save(output_preview_filename, icc_profile=rec2020_pq_profile)
    print(f"Image saved as {output_preview_filename} with Rec.2020 PQ ICC profile")
else:
    img_preview.save(output_preview_filename)
    print(f"Image saved as {output_preview_filename} (ICC profile not embedded)")


# --- Display P3 PQ HDR Export ---

# Create blank 16-bit image for Display P3 PQ HDR
img_p3_pq_array = np.zeros((img_height, img_width, 3), dtype=np.uint16)

# Process background for Display P3 PQ
bg_lin = srgb_to_linear(40)
bg_r_p3, bg_g_p3, bg_b_p3 = linear_srgb_to_display_p3(bg_lin, bg_lin, bg_lin)
bg_lum_r = max(0, bg_r_p3) * ref_white_nits_hdr
bg_lum_g = max(0, bg_g_p3) * ref_white_nits_hdr
bg_lum_b = max(0, bg_b_p3) * ref_white_nits_hdr
bg_pq_r = int(round(luminance_to_pq(bg_lum_r) * 65535))
bg_pq_g = int(round(luminance_to_pq(bg_lum_g) * 65535))
bg_pq_b = int(round(luminance_to_pq(bg_lum_b) * 65535))

# Initialize with background
img_p3_pq_array_final = np.full((img_height, img_width, 3), [bg_pq_r, bg_pq_g, bg_pq_b], dtype=np.uint16)

# Process patches for Display P3 PQ
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        srgb_color = patch_colors[idx]
        
        # 1. sRGB to Linear sRGB
        lin_r = srgb_to_linear(srgb_color[0])
        lin_g = srgb_to_linear(srgb_color[1])
        lin_b = srgb_to_linear(srgb_color[2])
        
        # 2. Linear sRGB to Linear Display P3
        p3_r, p3_g, p3_b = linear_srgb_to_display_p3(lin_r, lin_g, lin_b)
        
        # Clamp negative values
        p3_r = max(0, p3_r)
        p3_g = max(0, p3_g)
        p3_b = max(0, p3_b)
        
        # 3. Scale to Luminance (Reference White 203 nits)
        lum_r = p3_r * ref_white_nits_hdr
        lum_g = p3_g * ref_white_nits_hdr
        lum_b = p3_b * ref_white_nits_hdr
        
        # 4. Apply PQ OETF
        pq_r = luminance_to_pq(lum_r)
        pq_g = luminance_to_pq(lum_g)
        pq_b = luminance_to_pq(lum_b)
        
        # 5. Quantize to 16-bit
        val_r = int(round(pq_r * 65535))
        val_g = int(round(pq_g * 65535))
        val_b = int(round(pq_b * 65535))
        
        # Draw on array
        x0 = spacing + j * (patch_width + spacing)
        y0 = spacing + i * (patch_height + spacing)
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        
        img_p3_pq_array_final[y0:y1, x0:x1] = [val_r, val_g, val_b]

# Create separate copies for TIFF and PNG preview exports
tiff_p3_pq_array = img_p3_pq_array_final.copy()
png_p3_pq_preview_array = img_p3_pq_array_final.copy()

# Add filename text to Display P3 PQ TIFF
output_p3_pq_tiff = "macbeth_colorchecker_HDR_W203_DisplayP3_PQ_NF.tif"
text_mask_p3_pq_tiff = create_text_mask(img_width, img_height, output_p3_pq_tiff)
text_color_p3_pq = int(0.6 * 65535)

for y in range(img_height):
    for x in range(img_width):
        if text_mask_p3_pq_tiff[y, x]:
            tiff_p3_pq_array[y, x] = [text_color_p3_pq, text_color_p3_pq, text_color_p3_pq]

# Save Display P3 PQ TIFF (16-bit) using ImageMagick
r_p3_pq_filename = "temp_p3_pq_R.tiff"
g_p3_pq_filename = "temp_p3_pq_G.tiff"
b_p3_pq_filename = "temp_p3_pq_B.tiff"

r_ch_p3_pq = (Image.fromarray(tiff_p3_pq_array[:,:,0])).convert('I;16')
g_ch_p3_pq = (Image.fromarray(tiff_p3_pq_array[:,:,1])).convert('I;16')
b_ch_p3_pq = (Image.fromarray(tiff_p3_pq_array[:,:,2])).convert('I;16')

r_ch_p3_pq.save(r_p3_pq_filename)
g_ch_p3_pq.save(g_p3_pq_filename)
b_ch_p3_pq.save(b_p3_pq_filename)

try:
    cmd = ["magick", r_p3_pq_filename, g_p3_pq_filename, b_p3_pq_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_DISPLAY_P3_PQ, output_p3_pq_tiff]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        cmd = ["convert", r_p3_pq_filename, g_p3_pq_filename, b_p3_pq_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_DISPLAY_P3_PQ, output_p3_pq_tiff]
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully created {output_p3_pq_tiff}")
        os.remove(r_p3_pq_filename)
        os.remove(g_p3_pq_filename)
        os.remove(b_p3_pq_filename)
    else:
        print(f"Error combining Display P3 PQ TIFFs: {result.stderr}")
except Exception as e:
    print(f"Failed to run ImageMagick for Display P3 PQ TIFF: {e}")

# Generate 8-bit PNG preview from Display P3 PQ
output_p3_pq_preview = "macbeth_colorchecker_HDR_W203_DisplayP3_PQ_NF.png"

# Add filename text to PNG preview array
text_mask_p3_pq_preview = create_text_mask(img_width, img_height, output_p3_pq_preview)
text_color_p3_pq_preview = int(0.6 * 65535)

for y in range(img_height):
    for x in range(img_width):
        if text_mask_p3_pq_preview[y, x]:
            png_p3_pq_preview_array[y, x] = [text_color_p3_pq_preview, text_color_p3_pq_preview, text_color_p3_pq_preview]

# Convert to 8-bit for PNG preview
img_p3_pq_8bit = (png_p3_pq_preview_array / 256).astype(np.uint8)
img_p3_pq_preview = Image.fromarray(img_p3_pq_8bit, mode='RGB')

p3_pq_profile = create_display_p3_pq_profile()
if p3_pq_profile:
    img_p3_pq_preview.save(output_p3_pq_preview, icc_profile=p3_pq_profile)
    print(f"Image saved as {output_p3_pq_preview} with Display P3 PQ ICC profile")
else:
    img_p3_pq_preview.save(output_p3_pq_preview)
    print(f"Image saved as {output_p3_pq_preview} (ICC profile not embedded)")


print("\n=== Export Summary ===")
print("SDR Exports:")
print(f"  - {output_srgb_filename} (sRGB, 8-bit PNG)")
print(f"  - {output_p3_tiff} (Display P3, 16-bit TIFF)")
print(f"  - {output_p3_png} (Display P3, 8-bit PNG)")
print("\nHDR Exports (PQ):")
print(f"  - {output_tiff} (Rec.2020 PQ, 16-bit TIFF)")
print(f"  - {output_preview_filename} (Rec.2020 PQ, 8-bit PNG preview)")
print(f"  - {output_p3_pq_tiff} (Display P3 PQ, 16-bit TIFF)")
print(f"  - {output_p3_pq_preview} (Display P3 PQ, 8-bit PNG preview)")
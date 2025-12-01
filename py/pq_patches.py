import numpy as np
from PIL import Image, ImageDraw, ImageFont

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate PQ LED Wall TIFF patches.')
parser.add_argument('--max_lum', type=float, default=1000, help='Maximum luminance in nits (default: 1000)')
parser.add_argument('--num_patches', type=int, default=11, help='Number of patches (default: 11)')
args = parser.parse_args()

# Define LED Wall Max luminance
max_lum = args.max_lum

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

# Parameters
image_width, image_height = 3840, 2160
num_patches = args.num_patches

# Calculate safe area (Total 10% margin, 5% on each side)
margin_percentage = 0.05
safe_width = image_width * (1 - 2 * margin_percentage)
safe_height = image_height * (1 - 2 * margin_percentage)

# Calculate patch size to fit in safe width
# Total width = num_patches * patch_size + (num_patches - 1) * spacing
# Let spacing = 0.05 * patch_size
# safe_width = patch_size * (num_patches + (num_patches - 1) * 0.05)
patch_size = int(safe_width / (num_patches + (num_patches - 1) * 0.05))
spacing = int(patch_size * 0.05)

# --- Color Conversion Functions ---

def linear_srgb_to_rec2020(r, g, b):
    # Linear sRGB to XYZ (D65)
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    # XYZ to Linear Rec.2020 (D65)
    r_2020 = 1.7166511880 * X - 0.3556707838 * Y - 0.2533662814 * Z
    g_2020 = -0.6666843518 * X + 1.6164812366 * Y + 0.0157685458 * Z
    b_2020 = 0.0176398574 * X - 0.0427706533 * Y + 0.9421031254 * Z
    
    return r_2020, g_2020, b_2020

def linear_p3_to_rec2020(r, g, b):
    # Linear P3 (D65) to XYZ (D65)
    X = 0.4865709486 * r + 0.2656676931 * g + 0.1982172852 * b
    Y = 0.2289745640 * r + 0.6917393186 * g + 0.0792875976 * b
    Z = 0.0000000000 * r + 0.0451133814 * g + 1.0439443689 * b

    # XYZ to Linear Rec.2020 (D65)
    r_2020 = 1.7166511880 * X - 0.3556707838 * Y - 0.2533662814 * Z
    g_2020 = -0.6666843518 * X + 1.6164812366 * Y + 0.0157685458 * Z
    b_2020 = 0.0176398574 * X - 0.0427706533 * Y + 0.9421031254 * Z
    
    return r_2020, g_2020, b_2020

# --- Patch Generation ---

# 1. Grayscale Ramp
luminances_gray = [max_lum * (0.5)**i for i in range(num_patches)]
pq_values_gray = [luminance_to_pq(l) for l in luminances_gray]
pq_16bit_gray = []
for pq in pq_values_gray:
    val = int(round(pq * 65535))
    pq_16bit_gray.append((val, val, val)) # RGB tuple

# 2. Color Patches (R, G, B, C, M, Y)
# Base primaries in linear space (0-1)
primaries = [
    (1, 0, 0), # Red
    (0, 1, 0), # Green
    (0, 0, 1), # Blue
    (0, 1, 1), # Cyan
    (1, 0, 1), # Magenta
    (1, 1, 0)  # Yellow
]
primary_names = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow"]
ref_white_nits = 203.0
ref_white_nits_low = 100.0

# Helper function to calculate chromaticity coordinates
def rgb_to_chromaticity(r, g, b, color_space='rec2020'):
    """Convert RGB to chromaticity coordinates (x, y)"""
    if color_space == 'rec2020':
        # Rec.2020 linear to XYZ
        X = 0.6369580483 * r + 0.1446169036 * g + 0.1688809752 * b
        Y = 0.2627002120 * r + 0.6779980715 * g + 0.0593017165 * b
        Z = 0.0000000000 * r + 0.0280726930 * g + 1.0609850577 * b
    elif color_space == 'p3':
        # P3 (D65) linear to XYZ
        X = 0.4865709486 * r + 0.2656676931 * g + 0.1982172852 * b
        Y = 0.2289745640 * r + 0.6917393186 * g + 0.0792875976 * b
        Z = 0.0000000000 * r + 0.0451133814 * g + 1.0439443689 * b
    elif color_space == 'srgb':
        # sRGB linear to XYZ
        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    # Calculate chromaticity coordinates
    sum_XYZ = X + Y + Z
    if sum_XYZ == 0:
        return 0.0, 0.0
    x = X / sum_XYZ
    y = Y / sum_XYZ
    return x, y

# Calculate chromaticity coordinates for each primary in each color space
chromaticity_rec2020 = []
chromaticity_p3 = []
chromaticity_srgb = []

for r, g, b in primaries:
    chromaticity_rec2020.append(rgb_to_chromaticity(r, g, b, 'rec2020'))
    chromaticity_p3.append(rgb_to_chromaticity(r, g, b, 'p3'))
    chromaticity_srgb.append(rgb_to_chromaticity(r, g, b, 'srgb'))

# Set 1: Rec.2020 Primaries at 203 nits
pq_16bit_rec2020 = []
for r, g, b in primaries:
    # Already in Rec.2020 linear, just scale
    lum_r = r * ref_white_nits
    lum_g = g * ref_white_nits
    lum_b = b * ref_white_nits
    
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_rec2020.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))

# Set 2: P3 Primaries at 203 nits (in Rec.2020 container)
pq_16bit_p3 = []
for r, g, b in primaries:
    # Convert P3 linear to Rec.2020 linear
    r2020, g2020, b2020 = linear_p3_to_rec2020(r, g, b)
    
    # Clamp (shouldn't be needed for P3 inside 2020, but good practice)
    r2020 = max(0, r2020)
    g2020 = max(0, g2020)
    b2020 = max(0, b2020)
    
    lum_r = r2020 * ref_white_nits
    lum_g = g2020 * ref_white_nits
    lum_b = b2020 * ref_white_nits
    
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_p3.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))

# Set 3: sRGB Primaries at 203 nits (in Rec.2020 container)
pq_16bit_srgb = []
for r, g, b in primaries:
    # Convert sRGB linear to Rec.2020 linear
    r2020, g2020, b2020 = linear_srgb_to_rec2020(r, g, b)
    
    # Clamp
    r2020 = max(0, r2020)
    g2020 = max(0, g2020)
    b2020 = max(0, b2020)
    
    lum_r = r2020 * ref_white_nits
    lum_g = g2020 * ref_white_nits
    lum_b = b2020 * ref_white_nits
    
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_srgb.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))

# Set 4: Rec.2020 Primaries at 100 nits
pq_16bit_rec2020_low = []
for r, g, b in primaries:
    lum_r = r * ref_white_nits_low
    lum_g = g * ref_white_nits_low
    lum_b = b * ref_white_nits_low
    
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_rec2020_low.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))

# Set 5: P3 Primaries at 100 nits (in Rec.2020 container)
pq_16bit_p3_low = []
for r, g, b in primaries:
    r2020, g2020, b2020 = linear_p3_to_rec2020(r, g, b)
    
    r2020 = max(0, r2020)
    g2020 = max(0, g2020)
    b2020 = max(0, b2020)
    
    lum_r = r2020 * ref_white_nits_low
    lum_g = g2020 * ref_white_nits_low
    lum_b = b2020 * ref_white_nits_low
    
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_p3_low.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))

# Set 6: sRGB Primaries at 100 nits (in Rec.2020 container)
pq_16bit_srgb_low = []
for r, g, b in primaries:
    r2020, g2020, b2020 = linear_srgb_to_rec2020(r, g, b)
    
    r2020 = max(0, r2020)
    g2020 = max(0, g2020)
    b2020 = max(0, b2020)
    
    lum_r = r2020 * ref_white_nits_low
    lum_g = g2020 * ref_white_nits_low
    lum_b = b2020 * ref_white_nits_low
    
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_srgb_low.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))

# --- Real-Life Color Patches (P3 Linear RGB at 203 nits) ---
# These are normalized P3 RGB values (0-1 range)
reallife_colors_p3 = [
    (0.750, 0.050, 0.080),  # Deep Red Rose Petals
    (0.900, 0.200, 0.250),  # Ripe Strawberry
    (0.950, 0.400, 0.100),  # Burnt Orange Sunset
    (0.980, 0.650, 0.050),  # Marigold Bloom
    (0.900, 0.850, 0.150),  # Sunflower Yellow
    (0.300, 0.700, 0.150),  # Fresh Lime Zest
    (0.050, 0.550, 0.250),  # Forest Canopy
    (0.000, 0.450, 0.500),  # Deep Turquoise Sea
    (0.100, 0.400, 0.950),  # Electric Blue Sky
    (0.450, 0.200, 0.800),  # Royal Iris Flower
    (0.550, 0.200, 0.600),  # Amethyst Crystal
]

reallife_names = [
    "Deep Red Rose Petals",
    "Ripe Strawberry",
    "Burnt Orange Sunset",
    "Marigold Bloom",
    "Sunflower Yellow",
    "Fresh Lime Zest",
    "Forest Canopy",
    "Deep Turquoise Sea",
    "Electric Blue Sky",
    "Royal Iris Flower",
    "Amethyst Crystal",
]

def remove_srgb_gamma(val):
    """Remove sRGB gamma to get linear value"""
    if val <= 0.04045:
        return val / 12.92
    else:
        return ((val + 0.055) / 1.055) ** 2.4

# Convert P3 gamma-corrected to Rec.2020 PQ at 203 nits
pq_16bit_reallife = []
for r_p3_gamma, g_p3_gamma, b_p3_gamma in reallife_colors_p3:
    # Remove sRGB gamma to get linear P3 values
    r_p3_linear = remove_srgb_gamma(r_p3_gamma)
    g_p3_linear = remove_srgb_gamma(g_p3_gamma)
    b_p3_linear = remove_srgb_gamma(b_p3_gamma)
    
    # Convert P3 linear to Rec.2020 linear
    r2020, g2020, b2020 = linear_p3_to_rec2020(r_p3_linear, g_p3_linear, b_p3_linear)
    
    # Clamp
    r2020 = max(0, r2020)
    g2020 = max(0, g2020)
    b2020 = max(0, b2020)
    
    # Scale to luminance
    lum_r = r2020 * ref_white_nits
    lum_g = g2020 * ref_white_nits
    lum_b = b2020 * ref_white_nits
    
    # Apply PQ
    pq_r = luminance_to_pq(lum_r)
    pq_g = luminance_to_pq(lum_g)
    pq_b = luminance_to_pq(lum_b)
    
    pq_16bit_reallife.append((
        int(round(pq_r * 65535)),
        int(round(pq_g * 65535)),
        int(round(pq_b * 65535))
    ))


# --- Layout & Drawing ---

# Create blank 16-bit image (RGB)
img_array = np.zeros((image_height, image_width, 3), dtype=np.uint16)

# Calculate layout
total_patch_width_gray = num_patches * patch_size + (num_patches - 1) * spacing
left_margin_gray = (image_width - total_patch_width_gray) // 2

# Group Layout
# Each group is 3x2 patches
# Group width = 3 * patch_size + 2 * spacing
group_width = 3 * patch_size + 2 * spacing
# Group height = 2 * patch_size + spacing + text_spacing
vertical_spacing = int(patch_size * 0.8) # Space for text
group_height = 2 * patch_size + vertical_spacing # Patches + text space between rows? 
# Actually let's put text below each patch, so we need vertical space between rows inside the group
row_spacing = int(patch_size * 0.6) # Space between row 1 and row 2 labels
group_total_height = 2 * patch_size + row_spacing

# We have 3 groups side by side
# Calculate spacing to align outer edges with the grayscale ramp
# Total width of groups + spacing should equal total_patch_width_gray
# 3 * group_width + 2 * group_spacing = total_patch_width_gray
if total_patch_width_gray > 3 * group_width:
    group_spacing = (total_patch_width_gray - 3 * group_width) // 2
else:
    # Fallback if patches are too large (unlikely with current logic)
    group_spacing = int(patch_size * 0.5)

# Start X for groups should match Start X for grayscale ramp
start_x_groups = left_margin_gray

# Vertical positioning
# Row 1: Grayscale Ramp
# Gap
# Row 2: Groups (primaries)
# Gap
# Row 3: Real-life patches
reallife_patch_height = patch_size
reallife_text_space = int(patch_size * 0.4)  # Space for text below real-life patches
total_content_height = patch_size + vertical_spacing + group_total_height + vertical_spacing + reallife_patch_height + reallife_text_space
start_y = (image_height - total_content_height) // 2

# Helper function to draw text
def draw_text_on_array(img_arr, x, y, text, font_size=40):
    # Create temp image for text
    temp_img = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(temp_img)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((x, y), text, font=font, fill=255)
    mask = np.array(temp_img) > 0
    
    # Apply text at PQ 0.6 where mask is true
    text_value = int(round(0.6 * 65535))  # PQ 0.6 = 39321
    img_arr[mask] = [text_value, text_value, text_value]

# 1. Draw Grayscale Ramp
row_y = start_y
for i, val_tuple in enumerate(pq_16bit_gray):
    x = left_margin_gray + i * (patch_size + spacing)
    img_array[row_y:row_y+patch_size, x:x+patch_size] = val_tuple
    
    # Text
    lum = luminances_gray[i]
    pq = pq_values_gray[i]
    text_lum = f"{lum:.1f} nits"
    text_pq = f"PQ: {pq:.3f}"
    
    draw_text_on_array(img_array, x, row_y + patch_size + 10, text_lum)
    draw_text_on_array(img_array, x, row_y + patch_size + 50, text_pq)

draw_text_on_array(img_array, left_margin_gray, row_y - 60, f"Grayscale Ramp (Rec.2020 PQ {int(max_lum)} nits)", 50)


# Helper to draw a 3x2 group
def draw_group(start_x, start_y, patches_high, patches_low, chromaticity_coords, title):
    # patches_high: R, G, B, C, M, Y at 203 nits
    # patches_low: R, G, B, C, M, Y at 100 nits
    # chromaticity_coords: list of (x, y) tuples for each primary
    # Row 1: R, G, B
    # Row 2: C, M, Y
    # Each patch is split vertically: left half = high nits, right half = low nits
    
    draw_text_on_array(img_array, start_x, start_y - 60, title, 50)
    
    # Row 1
    for i in range(3):
        val_tuple_high = patches_high[i]
        val_tuple_low = patches_low[i]
        x = start_x + i * (patch_size + spacing)
        y = start_y
        
        # Split patch in half vertically
        half_width = patch_size // 2
        # Left half: high nits (203)
        img_array[y:y+patch_size, x:x+half_width] = val_tuple_high
        # Right half: low nits (100)
        img_array[y:y+patch_size, x+half_width:x+patch_size] = val_tuple_low
        
        # Display chromaticity coordinates
        chrom_x, chrom_y = chromaticity_coords[i]
        draw_text_on_array(img_array, x, y + patch_size + 10, f"x: {chrom_x:.4f}")
        draw_text_on_array(img_array, x, y + patch_size + 50, f"y: {chrom_y:.4f}")

    # Row 2
    for i in range(3):
        val_tuple_high = patches_high[i+3]
        val_tuple_low = patches_low[i+3]
        x = start_x + i * (patch_size + spacing)
        y = start_y + patch_size + row_spacing
        
        # Split patch in half vertically
        half_width = patch_size // 2
        # Left half: high nits (203)
        img_array[y:y+patch_size, x:x+half_width] = val_tuple_high
        # Right half: low nits (100)
        img_array[y:y+patch_size, x+half_width:x+patch_size] = val_tuple_low
        
        # Display chromaticity coordinates
        chrom_x, chrom_y = chromaticity_coords[i+3]
        draw_text_on_array(img_array, x, y + patch_size + 10, f"x: {chrom_x:.4f}")
        draw_text_on_array(img_array, x, y + patch_size + 50, f"y: {chrom_y:.4f}")

# Draw Groups
groups_y = row_y + patch_size + vertical_spacing

# Group 1: Rec.2020 (Left)
draw_group(start_x_groups, groups_y, pq_16bit_rec2020, pq_16bit_rec2020_low, chromaticity_rec2020, f"Rec.2020 ({int(ref_white_nits)}/{int(ref_white_nits_low)} nits)")

# Group 2: P3 (Middle)
draw_group(start_x_groups + group_width + group_spacing, groups_y, pq_16bit_p3, pq_16bit_p3_low, chromaticity_p3, f"P3 in Rec.2020 ({int(ref_white_nits)}/{int(ref_white_nits_low)} nits)")

# Group 3: sRGB (Right)
draw_group(start_x_groups + 2 * (group_width + group_spacing), groups_y, pq_16bit_srgb, pq_16bit_srgb_low, chromaticity_srgb, f"sRGB in Rec.2020 ({int(ref_white_nits)}/{int(ref_white_nits_low)} nits)")


# --- Draw Real-Life Patches ---
reallife_y = groups_y + group_total_height + vertical_spacing

# Calculate layout for 11 patches
num_reallife = len(pq_16bit_reallife)
total_reallife_width = num_reallife * patch_size + (num_reallife - 1) * spacing
left_margin_reallife = (image_width - total_reallife_width) // 2

# Draw title
draw_text_on_array(img_array, left_margin_reallife, reallife_y - 60, f"Real-Life Colors (P3 in Rec.2020, {int(ref_white_nits)} nits)", 50)

# Draw patches
for i, (val_tuple, name) in enumerate(zip(pq_16bit_reallife, reallife_names)):
    x = left_margin_reallife + i * (patch_size + spacing)
    img_array[reallife_y:reallife_y+patch_size, x:x+patch_size] = val_tuple
    
    # Draw name below patch (smaller font to fit)
    draw_text_on_array(img_array, x, reallife_y + patch_size + 10, name, font_size=28)


# Step 7: Save as 16-bit TIFF
# Since we have RGB data now, we can't use mode='I;16' directly for the whole image in PIL.
# We will save separate channels and combine, similar to the other script.

output_tiff = f"pq_patches_UHD_{int(max_lum)}nits_{num_patches}patches.tiff"
r_img = (Image.fromarray(img_array[:,:,0])).convert('I;16')
g_img = (Image.fromarray(img_array[:,:,1])).convert('I;16')
b_img = (Image.fromarray(img_array[:,:,2])).convert('I;16')

r_filename = "temp_pq_R.tiff"
g_filename = "temp_pq_G.tiff"
b_filename = "temp_pq_B.tiff"

r_img.save(r_filename)
g_img.save(g_filename)
b_img.save(b_filename)

import subprocess
import os

# ICC Profile Path
ICC_PROFILE_REC2020_PQ = "/Library/Application Support/Adobe/Color/Profiles/HDR_UHD_ST2084.icc"

try:
    # Use ImageMagick to combine
    cmd = ["magick", r_filename, g_filename, b_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_tiff]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        cmd = ["convert", r_filename, g_filename, b_filename, "-combine", "-depth", "16", "-profile", ICC_PROFILE_REC2020_PQ, output_tiff]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
    if result.returncode == 0:
        print(f"Successfully created {output_tiff}")
        os.remove(r_filename)
        os.remove(g_filename)
        os.remove(b_filename)
    else:
        print(f"Error combining TIFFs: {result.stderr}")
        # Fallback: Save as PNG preview (8-bit)
        print("Saving 8-bit PNG preview instead...")
        img_8bit = (img_array / 256).astype(np.uint8)
        Image.fromarray(img_8bit, 'RGB').save(output_tiff.replace('.tiff', '.png'))

except Exception as e:
    print(f"Failed to run ImageMagick: {e}")
    # Fallback
    img_8bit = (img_array / 256).astype(np.uint8)
    Image.fromarray(img_8bit, 'RGB').save(output_tiff.replace('.tiff', '.png'))

# Also save a PNG preview with ICC profile
output_png = output_tiff.replace('.tiff', '.png')
img_8bit = (img_array / 256).astype(np.uint8)
img_png = Image.fromarray(img_8bit, 'RGB')

try:
    with open(ICC_PROFILE_REC2020_PQ, 'rb') as f:
        rec2020_pq_profile = f.read()
    img_png.save(output_png, icc_profile=rec2020_pq_profile)
    print(f"Saved PNG preview with ICC profile: {output_png}")
except Exception as e:
    print(f"Warning: Could not embed ICC profile in PNG: {e}")
    img_png.save(output_png)
    print(f"Saved PNG preview (no ICC): {output_png}")
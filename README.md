
🎨 Python Scripts for Color Chart Generation

This repository contains Python scripts designed to generate various color charts for a wide range of use cases, particularly in display calibration, video production, and image processing.

Scripts

The repository includes three main scripts:


1. macbeth_colorchecker.py

This script generates a display-referred Macbeth Color Checker chart in multiple color spaces:
sRGB
Display P3
Display P3 with Perceptual Quantizer (PQ)
Rec.2020 with Perceptual Quantizer (PQ)
Key Features:
Standard Dynamic Range (SDR): The reference white is set at 100 nits (relative).
High Dynamic Range (HDR): The reference white is set at 203 nits (absolute).
Color Profile Embedding: All exported files are embedded with their corresponding ICC profile.
Shutterstock

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/macbeth_colorchecker_SDR_W100_sRGB.png)

2. Macbeth_HelenAndJohn_UntoneMapped.py

This is a utility script for working with the ARRI Helen and John Color Chart data.
Functionality:
It copies the real-world color code values from the original Helen and John color chart.
It uses this data to create a full-frame EXR scene-referred color chart in UHD size.
The source data is processed in ACES 2065-1 using the AlexaLogC4 IDT within the ACES 2.0 framework.
The script also creates an untonemapped Rec.2020 + PQ chart in TIF and PNG formats. This output is primarily intended for Virtual Production LED Wall calibration checks.

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/macbeth_HelenAndJohn_UntoneMapped_Rec2020_PQ.png)

3. pq_patches.py

This script generates a specialized PQ-encoded color chart for checking HDR display performance.
Patch Content:
PQ Gray Patches: Generates a default of --num_patches 11 gray patches. The luminance starts at the Maximum Luminance of the target LED wall (default is --max_lum 1000 nits) and decreases by one stop for each subsequent patch.
Primary Patches: Includes RGBCMY primary color patches grouped by color space (sRGB/P3/Rec.2020). Each primary color patch is split side-by-side, displaying its representation at 203 nits and 100 nits white levels.
Wide Gamut Patches: The final row contains real-life color patches. This is used to verify the wide color gamut display capability of the output device.

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/pq_patches_UHD_1000nits_11patches.png)

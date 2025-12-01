# 🎨 Python Scripts for Color Chart Generation

This repository contains Python scripts designed to generate various color charts for a wide range of use cases.

The repository includes three main scripts:

💾 **1. macbeth_colorchecker.py**

>This script generates a display-referred Macbeth Color Checker chart in multiple color spaces:
>
>- ✅ sRGB (MacOS)
>- ✅ Display P3 (MacOS)
>- ✅ Display P3 with PQ (Customized profile)
>- ✅ Rec.2020 with PQ (Adobe)

Standard Dynamic Range (SDR): The reference white is set at 100 nits (relative).  
High Dynamic Range (HDR): The reference white is set at 203 nits (absolute).

<sub>

**Color Profile Embedding**: All exported files are embedded with their corresponding ICC profile.

```bash
ICC_PROFILE_SRGB = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
ICC_PROFILE_DISPLAY_P3 = "/System/Library/ColorSync/Profiles/Display P3.icc"
ICC_PROFILE_DISPLAY_P3_PQ = "/Library/Application Support/Adobe/Color/Profiles/P3_PQ.icc" (Customized HDR Display P3 with PQ)
ICC_PROFILE_REC2020_PQ = "/Library/Application Support/Adobe/Color/Profiles/HDR_UHD_ST2084.icc"
```

</sub>

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/macbeth_colorchecker_SDR_W100_sRGB.png)

💾 **2. Macbeth_HelenAndJohn_UntoneMapped.py**

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/py/macbeth_HelenAndJohn_AlexaLogC4_DisplayP3.png)

<https://www.arri.com/en/learn-help/learn-help-camera-system/camera-sample-footage-reference-image#tab-294302>

This script copies the real-world color code values from the original Helen and John color chart to create a full-frame EXR scene-referred color chart in UHD size.

The script also creates an untonemapped Rec.2020 + PQ + W203 chart in TIF and PNG formats, primarily intended for Virtual Production LED Wall calibration checks.

Color Pipeline : ARRI Alexa35 MXF (Helen and John) -> ACES 2.0 AlexLogC4 IDT -> ACES 2065-1 -> White 203 -> Rec.2020PQ

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/macbeth_HelenAndJohn_UntoneMapped_Rec2020_PQ_W203.png)

💾 **3. pq_patches.py**

This script generates a specialized PQ-encoded color chart for checking HDR display performance.

**Patch Content:**

- ✅ **PQ Gray Patches**: Generates a series of gray patches (default is --num_patches 11) . The luminance starts at the Maximum Luminance of the target LED wall (default is --max_lum 1000 nits) and decreases by one stop for each subsequent patch.

- ✅ **Primary Patches**: Includes RGBCMY primary color patches grouped by color space (Rec.2020/P3/sRGB). Each primary color patch is split side-by-side, displaying its representation at 203 nits and 100 nits white levels.

- ✅ **A Series of Colorful Patches in Visible Spectrum Order**: The final row contains real-life color patches. This is used to verify the wide color gamut display capability of the output device.

![Alt text](https://github.com/KritsadaKaewmani/ColorChart_UHD_HDR/blob/main/pq_patches_UHD_1000nits_11patches.png)

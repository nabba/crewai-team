# Estonia Annual Deforestation & Forest Age Maps

Generate annual deforestation and forest age maps for Estonia from 2012 to present using Hansen Global Forest Change data via Google Earth Engine.

## Overview

This script creates two types of maps for each year:

1. **Binary Deforestation Map**: `estonia_defor_YYYY.tif`
   - Value 1: Forest loss occurred in this specific year
   - Value 0: No forest loss this year
   - Masked to pixels with ≥30% tree cover in 2000

2. **Forest Age Map**: `estonia_age_YYYY.tif`
   - Continuous values representing years since last disturbance
   - Never-disturbed forest: age = years since 2000 baseline
   - Disturbed forest: age = target year - year of last loss

## Age Calculation Logic (CORRECTED)

The Hansen dataset's `lossyear` band encodes loss timing:
- Value `0`: No forest loss recorded (2000-2023)
- Value `1`: First loss year (2001)
- Value `20`: 20th loss year (2020)

**Corrected formula** (avoids age=0 for undisturbed forest):

```
For a given target_year Y:

If lossyear == 0:
    age = Y - 2000  # Undisturbed since baseline
Else:
    loss_actual_year = 2000 + lossyear
    age = Y - loss_actual_year  # Years since last disturbance
```

**Example:**
- Pixel with `lossyear=0` in year 2023 → age = 23 years (not 0!)
- Pixel with `lossyear=5` (lost in 2005) in year 2023 → age = 18 years
- Pixel with `lossyear=12` (lost in 2012) in year 2023 → age = 11 years

## Installation & Setup

### Prerequisites

1. Google Earth Engine account: https://earthengine.google.com/
2. Python 3.8+
3. Earth Engine Python client:

```bash
pip install earthengine-api
```

### Authentication

Authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

This will open a browser window to authorize access to your Google account.

## Usage

### Basic Execution

```python
python skills/estonia_forest_age_maps/estonia_forest_age_maps.py
```

### As Module

```python
from skills.estonia_forest_age_maps.estonia_forest_age_maps import ee, summary

# After running the script, access results:
print(f"Total years processed: {summary['total_years_processed']}")
print(f"Total deforestation 2012-2023: {sum(y['deforestation_ha'] for y in summary['yearly_stats']):.2f} ha")
```

## Configuration Parameters

Edit the script's configuration section to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HANSEN_DATASET` | `'UMD/hansen/global_forest_change_2023_v1_11'` | Hansen GFC dataset ID |
| `TREE_COVER_THRESHOLD` | 30 | Minimum tree cover % to be considered forest |
| `START_YEAR` | 2012 | First year to analyze |
| `END_YEAR` | 2023 | Last year to analyze |
| `BASELINE_YEAR` | 2000 | Hansen baseline year |

## Outputs

### Google Drive Exports

The script prepares exports to your Google Drive:

- **Deforestation**: `estonia_defor_YYYY.tif`
- **Forest Age**: `estonia_age_YYYY.tif`

Each GeoTIFF is:
- Clipped to Estonia boundary
- 30m resolution (Hansen native)
- Single-band raster
- GeoTIFF format

### Console Summary

Returns a dictionary with:

```python
{
    'country': 'Estonia',
    'dataset': 'UMD/hansen/global_forest_change_2023_v1_11',
    'start_year': 2012,
    'end_year': 2023,
    'tree_cover_threshold_pct': 30,
    'yearly_stats': [
        {'year': 2012, 'deforestation_ha': 1234.56},
        {'year': 2013, 'deforestation_ha': 987.65},
        ...
    ],
    'loss_histogram': {...},
    'loss_area_histogram': {...}
}
```

## Datasets Used

### Hansen Global Forest Change (GFC)
- **Source**: University of Maryland / Google
- **ID**: `UMD/hansen/global_forest_change_2023_v1_11`
- **Resolution**: 30m
- **Coverage**: Global, 2000-present
- **Bands used**:
  - `lossyear`: Year of forest loss (0 = no loss, 1-20 = 2001-2020)
  - `treecover2000`: Percent tree cover in year 2000
  - `loss`: Binary forest loss mask

### LSIB 2017 (Large Scale International Boundary)
- **Source**: U.S. Department of State
- **ID**: `USDOS/LSIB/2017`
- **Property**: `country_na` = 'Estonia'

## Performance Notes

- Uses **single `.getInfo()` call** for statistics (via `ee.Reducer.frequencyHistogram`)
- Exports run asynchronously in Google Earth Engine backend
- Processing time: ~1-2 minutes for Estonia (small country)
- Export time: Varies by queue load, typically 5-15 minutes per file

## Limitations

1. **Data Lag**: Hansen GFC has ~1-2 year lag; current year may be unavailable
2. **Minimum Mapping Unit**: ~0.09 ha (30m pixels); small clearings may be missed
3. **Definition of Forest**: Based on tree cover threshold (default 30%)
4. **No Regrowth Detection**: Tracks loss only, not regrowth or secondary forest
5. **Age Proxy**: Age is time since last disturbance, not biological age

## Troubleshooting

### "EEException: User memory limit exceeded"
- Reduce the number of years processed at once
- Process smaller geographic regions

### "Image.computePixels: Error generating image"
- Verify Hansen dataset is accessible
- Check that Estonia geometry is valid

### Exports not appearing in Drive
- Check Google Earth Engine Tasks tab: https://code.earthengine.google.com/
- Verify your Drive has sufficient storage
- Ensure exports have completed (green checkmark)

### "Country 'Estonia' not found"
- Verify LSIB 2017 property is `country_na` (not `country_name`)
- Check spelling is exactly 'Estonia' (case-sensitive)

## Validation

To verify the age calculation logic is correct:

```python
# Test case 1: Undisturbed forest (lossyear=0)
# Should get age = target_year - 2000, not 0

# Test case 2: Forest lost in 2012 (lossyear=12)
# For target_year=2023, should get age = 11

# Test case 3: Forest lost in 2020 (lossyear=20)
# For target_year=2023, should get age = 3
```

## References

- Hansen, M.C. et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*, 342(6160), 850-853.
- Hansen GFC Documentation: https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2023_v1_11
- Google Earth Engine Python API: https://developers.google.com/earth-engine/guides/python_install

## License

This script uses public datasets. Hansen GFC is available under CC BY 4.0. LSIB is public domain.

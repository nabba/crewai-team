---
aliases:
- estonia forest maps
author: idle_scheduler.wiki_synthesis
confidence: medium
created_at: '2026-05-06T12:05:03Z'
date: '2026-05-06'
related: []
relationships: []
section: meta
source: workspace/skills/estonia_forest_maps.md
status: active
tags:
- self-improvement
- skills
- auto-synthesised
title: 'Skill: Generating Estonia Deforestation & Forest Age Maps via Hansen GFC'
updated_at: '2026-05-06T12:05:03Z'
version: 1
---

# Skill: Generating Estonia Deforestation & Forest Age Maps via Hansen GFC

## When to Use
Use this skill when you need to generate annual tree cover loss maps or forest age proxy maps for Estonia using the Hansen Global Forest Change (GFC) dataset via Google Earth Engine.

## Prerequisites
- Google Earth Engine account with authentication
- For Python: `earthengine-api` package installed (`pip install earthengine-api`)
- For JavaScript: Access to GEE Code Editor at https://code.earthengine.google.com/

## Method 1: GEE Code Editor (JavaScript - Recommended)

### Steps
1. Open https://code.earthengine.google.com/
2. Copy the contents of `output/ee_estonia_forest_maps.js`
3. Paste into the Code Editor
4. Click "Run" to execute
5. Export tasks will appear in the "Tasks" tab (right sidebar)
6. Click "Run" on each task to initiate export to your Google Drive
7. Wait for completion and download GeoTIFF files

### Output Files
- `estonia_deforestation_YYYY.tif` - Binary map: 1=loss in year YYYY, 0=no loss
- `estonia_forest_age_YYYY.tif` - Years since last disturbance (0=no loss)
- `estonia_forest_summary.csv` - Summary statistics

## Method 2: Python Local Execution

### Steps
1. Authenticate with Google Earth Engine:
   ```bash
   earthengine authenticate
   ```
2. Run the Python script:
   ```bash
   python output/estonia_forest_maps_complete.py
   ```
3. Export tasks will be queued to your Google Drive
4. Download completed GeoTIFF files

### Key Code Patterns

#### Define Estonia Geometry
```javascript
var estonia = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq('country_na', 'Estonia'));
```

```python
estonia = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
    ee.Filter.eq('country_na', 'Estonia')
)
```

#### Load Hansen GFC Dataset
```javascript
var gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11');
var lossYear = gfc.select('lossyear');
var dataMask = gfc.select('datamask');
```

#### Create Annual Deforestation Map
```javascript
var year = 2012;
var hansenYear = year - 2000;  // Hansen encodes 2001 as 1
var deforestationMap = lossYear.eq(hansenYear)
  .rename('deforestation')
  .updateMask(dataMask.eq(1));
```

#### Create Annual Forest Age Map
```javascript
var lossYearActual = lossYear.add(2000);
var forestAgeMap = ee.Image(year)
  .subtract(lossYearActual)
  .where(lossYear.eq(0), 0)
  .max(0)
  .rename('forest_age')
  .updateMask(dataMask.eq(1));
```

#### Export to Google Drive
```javascript
Export.image.toDrive({
  image: deforestationMap,
  description: 'estonia_deforestation_2012',
  fileNamePrefix: 'estonia_deforestation_2012',
  region: estonia.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
```

## Dataset Details

### Hansen Global Forest Change (2023 v1.11)
- **Source**: UMD/hansen/global_forest_change_2023_v1_11
- **Spatial Resolution**: 30 meters (Landsat)
- **Temporal Range**: 2000-2023
- **Bands**:
  - `treecover2000`: Percent tree canopy cover in year 2000 (0-100)
  - `lossyear`: Year of gross forest loss event (1=2001 to 23=2023)
  - `loss`: Binary forest loss mask
  - `gain`: Binary forest gain mask (2000-2012)
  - `datamask`: Data quality mask (1=valid data)

### Estonia Geometry
- **Source**: USDOS/LSIB_SIMPLE/2017 (U.S. Department of State LSIB)
- **Filter**: `country_na = 'Estonia'`
- **Area**: ~45,000 km²
- **Forest Cover**: ~50% of land area

## Output Specifications

### Deforestation Maps (`estonia_deforestation_YYYY.tif`)
- **Format**: GeoTIFF
- **Coordinate System**: EPSG:4326 (WGS84)
- **Resolution**: 30 meters
- **Data Type**: Integer
- **Values**: 
  - 1: Forest loss detected in year YYYY
  - 0: No forest loss (masked)

### Forest Age Maps (`estonia_forest_age_YYYY.tif`)
- **Format**: GeoTIFF
- **Coordinate System**: EPSG:4326 (WGS84)
- **Resolution**: 30 meters
- **Data Type**: Integer
- **Values**:
  - 0: No forest loss observed (intact forest)
  - 1, 2, 3, ...: Years since last disturbance

## Performance Notes

1. **Server-Side Aggregation**: Use `ee.Reducer.frequencyHistogram()` with a single `.getInfo()` call to avoid timeouts
2. **Task Management**: Large exports may take several minutes each; monitor in Tasks tab
3. **Quota Limits**: GEE has export quotas; batch exports sequentially if needed
4. **Memory**: Exporting all years (24 maps) may exceed storage limits; process in batches if needed

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Authentication error | Run `earthengine authenticate` in terminal |
| Export tasks not starting | Check GEE quota; retry after existing tasks complete |
| "Computation timed out" | Reduce region size or use server-side aggregation |
| "User memory limit exceeded" | Use `.reduceRegion()` with `maxPixels=1e13` |
| Missing data in output | Check `dataMask` band; only valid pixels are included |

## References
- Hansen et al. (2013): "High-Resolution Global Maps of 21st-Century Forest Cover Change"
- GEE Documentation: https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_v1_11

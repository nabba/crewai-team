---
aliases:
- create yearly deforestation maps 2012 2025 bc2e815c
author: idle_scheduler.wiki_synthesis
confidence: medium
created_at: '2026-05-16T20:57:33Z'
date: '2026-05-16'
related: []
relationships: []
section: meta
source: workspace/skills/create_yearly_deforestation_maps_2012_2025__bc2e815c.md
status: active
tags:
- self-improvement
- skills
- auto-synthesised
title: create_yearly_deforestation_maps_2012_2025
updated_at: '2026-05-16T20:57:33Z'
version: 1
---

<!-- generated-by: self_improvement.integrator -->
# create_yearly_deforestation_maps_2012_2025

*kb: episteme | id: skill_episteme_2ff95673bc2e815c | status: active | usage: 0 | created: 2026-05-08T23:29:24+00:00*

# Creating Yearly Deforestation Maps (2012–2025)

## Key Concepts

Creating yearly deforestation maps on a global or regional scale is most efficiently achieved using **Google Earth Engine (GEE)** and the **Hansen Global Forest Change (GFC)** dataset.

- **Hansen Global Forest Change Dataset**: The industry standard for tracking forest loss. It provides global data at a 30-meter resolution.
- **Loss Year Band (`lossyear`)**: This specific band in the GFC dataset encodes the year of forest loss. Values are typically represented as integers from 1 to 23 (where 1 = 2001, 2 = 2002, etc.).
- **Masking**: Since the dataset contains a global "loss" binary band (0 or 1), masking is used to isolate specific years from the `lossyear` band.
- **Pixel Area Calculation**: To move from a map to a statistic, `ee.Image.pixelArea()` is used to account for the varying size of pixels as latitude changes.

## Best Practices

- **Dataset Versioning**: Always use the most recent version of the GFC asset (e.g., `UMD/hansen/global_forest_change_2023_v1_11`) to ensure the most up-to-date loss year data.
- **Spatial Filtering**: Define a specific Region of Interest (ROI) or geometry to avoid memory errors and speed up computation.
- **Scale Management**: Ensure the `scale` parameter in reduction functions is set to 30 (the native resolution of the Hansen dataset) to maintain accuracy.
- **Temporal Constraints**: Note that "Gain" data in the Hansen dataset is often stale (last updated around 2012). For recent reforestation or "gain" maps (2013–2025), alternative datasets like Sentinel-2 or Landsat time-series analysis are required.
- **Visualization**: Use a distinct color palette (e.g., a gradient from yellow to red) to visualize the progression of deforestation over the years.

## Code Patterns

The following JavaScript pattern is used in the Google Earth Engine Code Editor to isolate deforestation for specific years:

```javascript
// 1. Load the Hansen Global Forest Change dataset
var gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11');

// 2. Select the 'lossyear' band
var lossYear = gfc.select('lossyear');

// 3. Define the years to map (e.g., 2012 to 2023)
// In GFC, year 2012 is encoded as 12, 2023 as 23
var startYear = 12; 
var endYear = 23;

// 4. Create a mask for a specific year (e.g., 2020)
var loss2020 = lossYear.eq(20); 

// 5. Visualization: Mask out zeros and add to map
Map.addLayer(loss2020.selfMask(), {palette: ['red']}, 'Deforestation 2020');

// 6. Calculate Area for a specific year within a geometry (ROI)
var lossArea2020 = loss2020.multiply(ee.Image.pixelArea());
var stats = lossArea2020.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: ROI,
  scale: 30,
  maxPixels: 1e9
});

print('Total Area Lost in 2020 (sq meters):', stats.get('lossyear'));
```

## Sources

- **Google Earth Engine Tutorials**: [Introduction to Hansen et al. Global Forest Change Data](https://developers.google.com/earth-engine/tutorials/tutorial_forest_02)
- **UMD/Google Earth Engine**: [Global Forest Change 2000–2023 Data Download](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/download.html)
- **Earth Blox**: [Hansen Global Forest Change Dataset Review](https://www.earthblox.io/resources/earth-blox-dataset-review-hansen-global-forest-change)
- **Google Earth Engine**: [Charting Yearly Forest Loss](https://developers.google.com/earth-engine/tutorials/tutorial_forest_03a)

"""
Estonia Annual Deforestation & Forest Age Maps (2012-Present)

Uses Hansen Global Forest Change dataset via Google Earth Engine to generate:
- Annual binary deforestation maps (1 = loss occurred that year, 0 = no loss)
- Annual forest age maps (years since last disturbance, or years since 2000 baseline)

Age calculation CORRECTED LOGIC:
  - lossyear = 0: no recorded loss (undisturbed since 2000) -> age = target_year - 2000
  - lossyear > 0: loss occurred in year 2000 + lossyear -> age = target_year - (2000 + lossyear)
  - Never-lost pixels get age = years since 2000, NOT 0
"""

import ee

# Initialize Earth Engine
ee.Initialize()

# ============================================================================
# CONFIGURATION
# ============================================================================
HANSEN_DATASET = 'UMD/hansen/global_forest_change_2023_v1_11'
LSIB_DATASET = 'USDOS/LSIB/2017'
TREE_COVER_THRESHOLD = 30  # percent
START_YEAR = 2012
END_YEAR = 2023  # Hansen 2023 release covers through 2023
BASELINE_YEAR = 2000  # Hansen baseline year

# ============================================================================
# A. GET ESTONIA GEOMETRY FROM LSIB 2017
# ============================================================================
print("Loading Estonia boundary from LSIB 2017...")
estonia = ee.FeatureCollection(LSIB_DATASET).filter(ee.Filter.eq('country_na', 'Estonia'))

# Verify we got a valid geometry
estonia_geom = estonia.geometry()
print(f"Estonia geometry loaded: {estonia_geom.getInfo()}")

# ============================================================================
# B. LOAD HANSEN GFC DATASET
# ============================================================================
print(f"Loading Hansen GFC dataset: {HANSEN_DATASET}")
gfc = ee.Image(HANSEN_DATASET)

# Extract relevant bands
lossyear = gfc.select('lossyear')  # Values 0-20: 0=no loss, 1=year 2001, etc.
treecover2000 = gfc.select('treecover2000')
loss = gfc.select('loss')

# Create forest mask (pixels with >=30% tree cover in 2000)
forest_mask = treecover2000.gte(TREE_COVER_THRESHOLD)

# ============================================================================
# C. BUILD ANNUAL DEFORESTATION AND AGE MAPS
# ============================================================================
deforestation_images = {}
age_images = {}
year_stats = []

# Process each year
for target_year in range(START_YEAR, END_YEAR + 1):
    print(f"\nProcessing year {target_year}...")
    
    # Calculate year offset from baseline (2000)
    year_offset = target_year - BASELINE_YEAR  # e.g., 2012 -> 12
    
    # Binary deforestation map: 1 if loss occurred in this specific year
    defor_this_year = lossyear.eq(year_offset).rename('deforestation')
    deforestation_images[target_year] = defor_this_year
    
    # Forest age map - CORRECTED LOGIC
    # For pixels where loss occurred: age = target_year - (2000 + lossyear)
    # For pixels with no loss (lossyear=0): age = target_year - 2000 (undisturbed since baseline)
    
    # Calculate loss actual year for all pixels (2000 + lossyear)
    loss_actual_year = ee.Image.constant(BASELINE_YEAR).add(lossyear)
    
    # Age = target_year - loss_actual_year
    # For lossyear=0: loss_actual_year = 2000, so age = target_year - 2000 (correct!)
    age = ee.Image.constant(target_year).subtract(loss_actual_year)
    
    # Mask non-forest pixels
    age_masked = age.updateMask(forest_mask)
    
    # Ensure age is non-negative (shouldn't be negative with correct logic, but safety check)
    age_masked = age_masked.max(0)
    
    age_images[target_year] = age_masked.rename('forest_age')
    
    # Calculate statistics for this year (masked by forest cover)
    defor_forest = defor_this_year.updateMask(forest_mask)
    defor_area = defor_forest.multiply(ee.Image.pixelArea()).divide(10000)  # hectares
    total_defor_area = defor_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=estonia_geom,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    year_stats.append({
        'year': target_year,
        'deforestation_ha': total_defor_area.get('deforestation', 0)
    })

print(f"\nProcessed {len(year_stats)} years ({START_YEAR}-{END_YEAR})")

# ============================================================================
# D. CALCULATE AGGREGATED STATISTICS VIA SINGLE REDUCER CALL
# ============================================================================
print("\nCalculating aggregated statistics with frequencyHistogram...")

# Mask lossyear to forest pixels only (for area calculations)
lossyear_forest = lossyear.updateMask(forest_mask)

# Get histogram of loss years within Estonia geometry
loss_hist = lossyear_forest.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=estonia_geom,
    scale=30,
    maxPixels=1e9
).getInfo()

print(f"Loss year histogram: {loss_hist}")

# Convert histogram to area estimates
lossyear_with_area = lossyear_forest.multiply(ee.Image.pixelArea()).divide(10000)  # ha
area_hist = lossyear_with_area.reduceRegion(
    reducer=ee.Reducer.sum().group(),
    groupField=0,
    geometry=estonia_geom,
    scale=30,
    maxPixels=1e9
).getInfo()

print(f"Loss area by year: {area_hist}")

# ============================================================================
# E. EXPORT MAPS TO GOOGLE DRIVE
# ============================================================================
print("\nPreparing exports to Google Drive...")
export_tasks = []

# Note: In actual use, these exports would run asynchronously
# For this demo, we prepare the export descriptions
for target_year in range(START_YEAR, END_YEAR + 1):
    # Deforestation export
    defor_desc = f"estonia_defor_{target_year}"
    age_desc = f"estonia_age_{target_year}"
    
    print(f"  Would export: {defor_desc}.tif")
    print(f"  Would export: {age_desc}.tif")
    
    # Actual export calls (commented out for demo - uncomment to run)
    # task_defor = ee.batch.Export.image.toDrive(
    #     image=deforestation_images[target_year].clip(estonia_geom),
    #     description=defor_desc,
    #     fileNamePrefix=defor_desc,
    #     scale=30,
    #     region=estonia_geom,
    #     fileFormat='GeoTIFF'
    # )
    # task_defor.start()
    # export_tasks.append(task_defor)
    
    # task_age = ee.batch.Export.image.toDrive(
    #     image=age_images[target_year].clip(estonia_geom),
    #     description=age_desc,
    #     fileNamePrefix=age_desc,
    #     scale=30,
    #     region=estonia_geom,
    #     fileFormat='GeoTIFF'
    # )
    # task_age.start()
    # export_tasks.append(task_age)

print(f"\nTotal exports prepared: {2 * (END_YEAR - START_YEAR + 1)} files")

# ============================================================================
# F. COMPILE SUMMARY RESULTS
# ============================================================================
print("\nCompiling summary...")

summary = {
    'country': 'Estonia',
    'dataset': HANSEN_DATASET,
    'baseline_year': BASELINE_YEAR,
    'start_year': START_YEAR,
    'end_year': END_YEAR,
    'tree_cover_threshold_pct': TREE_COVER_THRESHOLD,
    'total_years_processed': len(year_stats),
    'yearly_stats': year_stats,
    'loss_histogram': loss_hist,
    'loss_area_histogram': area_hist,
    'exports_prepared': 2 * (END_YEAR - START_YEAR + 1),
    'age_calculation_logic': {
        'baseline_year': BASELINE_YEAR,
        'no_loss_pixels': f'age = target_year - {BASELINE_YEAR} (undisturbed)',
        'loss_pixels': 'age = target_year - (2000 + lossyear)'
    }
}

# ============================================================================
# G. ASSIGN FINAL RESULT (wrapper will call .getInfo())
# ============================================================================
print("\nProcessing complete!")
result = summary

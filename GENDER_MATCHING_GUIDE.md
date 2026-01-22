# Gender Matching Feature - ReActor V3

## Overview

ReActor V3 now includes intelligent gender detection and matching capabilities. This feature allows you to control face swapping based on gender, ensuring that only matching genders are swapped when desired.

## How It Works

The feature uses InsightFace's built-in gender detection model (`genderage.onnx`) which analyzes facial features to determine gender. This happens automatically during face detection with no additional processing overhead.

## Gender Matching Modes

### 1. **All (No Filter)** - Mode: `A`
- **Default behavior**: No gender filtering
- Swaps faces regardless of gender
- Use when gender doesn't matter or for creative effects

### 2. **Smart Match (Auto-detect)** - Mode: `S` ⭐ RECOMMENDED
- **Intelligent matching**: Automatically detects source face gender and only swaps with matching gender in target
- **Perfect for mixed scenes**: If target has both male and female, only the matching gender gets swapped
- **Example scenarios**:
  - Source: Female face → Target: Man + Woman → Only woman's face is swapped
  - Source: Male face → Target: Group of men → Swaps the selected male face
  - Source: Female face → Target: Only males → Returns error message, no swap

### 3. **Male Only** - Mode: `M`
- **Strict filtering**: Only processes male faces
- Both source and target must have male faces
- Useful when you only want to swap male faces in a scene with multiple people

### 4. **Female Only** - Mode: `F`
- **Strict filtering**: Only processes female faces
- Both source and target must have female faces
- Useful when you only want to swap female faces in a scene with multiple people

## Usage Examples

### Scenario 1: Mixed Gender Group Photo
**Setup:**
- Source: Single female portrait
- Target: Group photo with 2 women and 3 men
- Mode: **Smart Match (S)**

**Result:** Only the women's faces are considered for swapping. The men's faces are automatically ignored.

### Scenario 2: Female-Only Face Swap
**Setup:**
- Source: Female portrait
- Target: Image with multiple people (mixed genders)
- Mode: **Female Only (F)**

**Result:** Only female faces in both images are processed. Male faces are filtered out completely.

### Scenario 3: Traditional Swap (No Gender Filter)
**Setup:**
- Source: Any face
- Target: Any face
- Mode: **All (A)**

**Result:** Standard face swap behavior - first matching face by index, regardless of gender.

## UI Integration

The gender matching control appears in the ReActor V3 accordion in both img2img and txt2img tabs:

```
Gender Matching Mode
○ All (No Filter)
● Smart Match (Auto-detect)  ← Default
○ Male Only
○ Female Only
```

**Info text:** "S=Match source gender automatically, M/F=Filter specific gender"

## Technical Details

### Gender Detection
- Uses InsightFace's `genderage.onnx` model (already included with buffalo_l)
- Returns values: `0` (female) or `1` (male)
- Threshold: 0.5 (values ≥ 0.5 = male, < 0.5 = female)
- Unknown gender (`U`) is treated permissively to avoid over-filtering

### Face Filtering Process
1. Detect all faces in source and target images
2. Analyze gender for each detected face
3. Apply filtering based on selected mode
4. Select face by index from filtered list
5. Perform swap operation

### Error Handling
The system provides clear error messages when gender matching fails:
- "Error: No male face detected in source image"
- "Error: No female face found in target image to match source"
- "Error: No matching face in target to match source"

## Console Output

When gender matching is active, you'll see detailed logs:

```
[ReActor V3] Smart Match Mode: Source face gender = F
[ReActor V3] Filtering target faces to match source gender...
[ReActor V3] Face 0: Gender=M ✗
[ReActor V3] Face 1: Gender=F ✓
[ReActor V3] Face 2: Gender=F ✓
[ReActor V3] Found 2 matching face(s) in target
[ReActor V3] Swapping: Source (F) -> Target (F)
```

## Best Practices

### When to Use Smart Match (S)
✅ Mixed gender scenes (group photos, couples, family portraits)  
✅ When you want automatic gender-based selection  
✅ General use - it's the safest default  

### When to Use Male/Female Only (M/F)
✅ Batch processing same-gender faces  
✅ When you know both images contain only one gender  
✅ When you want to enforce gender consistency  

### When to Use All (A)
✅ Artistic/creative effects  
✅ Gender-neutral or ambiguous faces  
✅ When gender detection might be unreliable  

## Troubleshooting

### "No matching face found" errors
**Cause:** Gender filter too restrictive  
**Solution:** 
1. Switch to "All (A)" mode to see if faces are detected
2. Check console for gender detection results
3. Verify source and target actually contain matching genders

### Unexpected gender detection
**Cause:** Ambiguous facial features or styling  
**Solution:**
1. Use "All (A)" mode to bypass gender filtering
2. Try different source images with clearer gender markers
3. Check console logs to see detected gender values

### Wrong face selected after filtering
**Cause:** Face index refers to filtered list, not original  
**Solution:**
1. Adjust "Target Face Index" slider
2. Remember index 0 = first face of the filtered gender
3. Use Smart Match (S) instead of M/F for automatic selection

## API Usage

When calling ReActor V3 programmatically:

```python
result, status = engine.process(
    source_img=source_cv2,
    target_img=target_cv2,
    source_face_index=0,
    target_face_index=0,
    restore_model="GPEN-BFR-512.pth",
    gender_match='S'  # 'A', 'S', 'M', or 'F'
)
```

## Performance Impact

**Minimal:** Gender detection is part of the standard InsightFace analysis and adds no measurable overhead. The `genderage.onnx` model runs simultaneously with face detection.

## Compatibility

- ✅ Works with all GPEN restoration models
- ✅ Compatible with face index selection
- ✅ Works in both img2img and txt2img postprocessing
- ✅ No additional model downloads required
- ✅ Fully backward compatible (defaults to 'A' if not specified)

## Future Enhancements

Potential improvements:
- Age-based filtering
- Confidence threshold adjustment
- Multiple gender swaps in single operation
- Gender swap statistics in output

## Credits

Gender detection powered by InsightFace's buffalo_l model package.

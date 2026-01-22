# 🎨 Advanced Realism Guide - ReActor V3

## Maximum Realism Configuration

This guide shows how to configure ReActor V3 for **photorealistic** face swaps that are indistinguishable from real photos.

---

## 🎯 Quick Presets

### **Preset 1: Maximum Quality (Portraits)**
Perfect for close-up portraits, headshots, and professional photography:

| Setting | Value | Why |
|---------|-------|-----|
| **GPEN Model** | GPEN-BFR-1024.onnx | Ultra-high detail (2048px effective) |
| **Auto-Select Resolution** | ON | Smart model selection |
| **Resolution Threshold** | 256px | Aggressive 1024 usage |
| **Face Blend Strength** | 1.0 | Full swap for maximum detail |
| **Color Correction** | ON | Match lighting perfectly |
| **Upscale Factor** | 2x | Extract faces at 2048px |
| **Detection Threshold** | 0.6 | Quality control |

**Speed:** ~500ms per face | **VRAM:** 8GB+ recommended | **Quality:** ⭐⭐⭐⭐⭐

---

### **Preset 2: Balanced Quality (General Use)**
Best all-around settings for most scenarios:

| Setting | Value | Why |
|---------|-------|-----|
| **GPEN Model** | GPEN-BFR-512.onnx | Fast, excellent quality |
| **Auto-Select Resolution** | ON | Automatic optimization |
| **Resolution Threshold** | 384px | Balanced threshold |
| **Face Blend Strength** | 0.95 | Subtle natural blending |
| **Color Correction** | ON | Lighting match |
| **Upscale Factor** | 1x | Standard resolution |
| **Detection Threshold** | 0.5 | Standard detection |

**Speed:** ~200ms per face | **VRAM:** 6GB+ | **Quality:** ⭐⭐⭐⭐

---

### **Preset 3: Subtle Blend (Natural Merge)**
For when you want to blend features rather than full swap:

| Setting | Value | Why |
|---------|-------|-----|
| **GPEN Model** | GPEN-BFR-512.onnx | Good detail, fast |
| **Auto-Select Resolution** | ON | Auto optimization |
| **Face Blend Strength** | 0.70-0.85 | Merge features |
| **Color Correction** | ON | Essential for realism |
| **Upscale Factor** | 1x | Standard |
| **Detection Threshold** | 0.5 | Standard |

**Use case:** Create hybrid faces, subtle changes, "what if" scenarios

---

### **Preset 4: Speed Priority (Batch Processing)**
Fast processing for many images:

| Setting | Value | Why |
|---------|-------|-----|
| **GPEN Model** | GPEN-BFR-512.onnx | Fast inference |
| **Auto-Select Resolution** | ON | Smart selection |
| **Resolution Threshold** | 512px | Favor 512 |
| **Face Blend Strength** | 1.0 | Full swap |
| **Color Correction** | ON | Quick correction |
| **Upscale Factor** | 1x | No slowdown |
| **Detection Threshold** | 0.4 | More permissive |

**Speed:** ~180ms per face | **VRAM:** 4GB+ | **Quality:** ⭐⭐⭐⭐

---

## ⚙️ Advanced Parameter Guide

### 1. Face Blend Strength (0.0 - 1.0)

Controls how much of the swapped face is blended with the original.

| Value | Effect | Use Case |
|-------|--------|----------|
| **1.0** | Full replacement | Standard face swap, maximum accuracy |
| **0.90-0.95** | Subtle original features | Soften swap, keep some target traits |
| **0.70-0.85** | Hybrid blend | Merge features from both faces |
| **0.50-0.65** | Gentle merge | Subtle influence, mostly target |
| **0.0-0.40** | Minimal influence | Just hints of source face |

**💡 Pro Tip:** Use 0.92-0.97 for the most natural-looking full swaps that retain micro-expressions.

---

### 2. Face Detection Confidence (0.1 - 0.99)

Controls how confident the AI must be before detecting a face.

| Value | Effect | When to Use |
|-------|--------|-------------|
| **0.3-0.4** | Very permissive | Difficult angles, occluded faces, side profiles |
| **0.5** | Standard (default) | Normal frontal/near-frontal faces |
| **0.6-0.7** | Strict | Quality control, avoid false detections |
| **0.8+** | Very strict | High-confidence faces only, may miss valid faces |

**⚠️ Warning:** Too high (0.8+) may fail to detect valid faces. Too low (0.2-) may detect non-faces.

---

### 3. Color Correction

Automatically matches the swapped face's color/lighting to the target image.

| Setting | Effect | When to Use |
|---------|--------|-------------|
| **ON (Recommended)** | Swapped face matches target lighting | Always for realism |
| **OFF** | Preserves source face colors | Intentional color mismatch, artistic |

**How it works:**
- Analyzes LAB color space (perceptual color)
- Transfers mean and standard deviation
- Matches skin tone to target lighting conditions
- Preserves face structure while adjusting color

**💡 Pro Tip:** Always keep ON unless you specifically need to preserve source lighting.

---

### 4. Upscale Factor (1x or 2x)

Extracts faces at higher resolution before GPEN restoration.

| Value | Effect | VRAM | Speed | Quality |
|-------|--------|------|-------|---------|
| **1x** | Standard extraction | Low | Fast | ⭐⭐⭐⭐ |
| **2x** | Double resolution extraction | High | Slow | ⭐⭐⭐⭐⭐ |

**When to use 2x:**
- Large faces (>512px) in high-res images (4K+)
- With GPEN-1024 for maximum detail
- Professional photography, print quality
- When VRAM is not a concern (12GB+)

**Technical:** 
- 1x + GPEN-512 = ~512px faces
- 1x + GPEN-1024 = ~1024px faces
- 2x + GPEN-1024 = ~2048px faces (ultra-sharp)

---

### 5. Auto-Resolution Threshold (256-768px)

Face size above which GPEN-1024 is used (when Auto-Select is ON).

| Value | Effect | Use Case |
|-------|--------|----------|
| **256-320px** | Use 1024 aggressively | Maximum quality, slower |
| **384px (default)** | Balanced | Recommended default |
| **448-512px** | Favor 512 | Speed priority |
| **640-768px** | Almost always 512 | Fast batch processing |

**💡 Pro Tip:** Lower threshold (256-320) for portrait-focused workflows, higher (448-512) for group photos.

---

## 🎭 Scenario-Based Recommendations

### **Scenario: Professional Portrait Photography**

Goal: Indistinguishable from real photo, print quality

```
GPEN Model: GPEN-BFR-1024.onnx
Auto-Select: ON
Resolution Threshold: 256px
Blend Strength: 0.98
Color Correction: ON
Upscale Factor: 2x
Detection Threshold: 0.6
Gender Matching: Smart Match (S)
```

**Why:** Maximum quality at every step, strict quality control, ultra-sharp 2048px faces.

---

### **Scenario: Group Photo (Wedding/Event)**

Goal: Natural swaps, process multiple faces efficiently

```
GPEN Model: GPEN-BFR-512.onnx
Auto-Select: ON
Resolution Threshold: 384px
Blend Strength: 0.95
Color Correction: ON
Upscale Factor: 1x
Detection Threshold: 0.5
Gender Matching: Smart Match (S)
```

**Why:** Fast processing, good quality, auto-selects 1024 for larger faces.

---

### **Scenario: Social Media Content**

Goal: Fast, good quality, batch process many images

```
GPEN Model: GPEN-BFR-512.onnx
Auto-Select: ON
Resolution Threshold: 448px
Blend Strength: 1.0
Color Correction: ON
Upscale Factor: 1x
Detection Threshold: 0.4
Gender Matching: All (A)
```

**Why:** Speed optimized, permissive detection for various poses.

---

### **Scenario: Movie/VFX Face Replacement**

Goal: Seamless integration, color matched, ultra-realistic

```
GPEN Model: GPEN-BFR-1024.onnx
Auto-Select: OFF (manual 1024)
Resolution Threshold: N/A
Blend Strength: 0.92
Color Correction: ON
Upscale Factor: 2x
Detection Threshold: 0.5
Gender Matching: Smart Match (S)
```

**Why:** Slight blend (0.92) preserves micro-expressions, color correction matches scene lighting.

---

### **Scenario: Historical Photo Restoration**

Goal: Add faces to old photos, match vintage lighting

```
GPEN Model: GPEN-BFR-512.onnx
Auto-Select: ON
Resolution Threshold: 384px
Blend Strength: 0.88
Color Correction: ON
Upscale Factor: 1x
Detection Threshold: 0.4
Gender Matching: Smart Match (S)
```

**Why:** Lower blend preserves photo aging effects, permissive detection for old photos, color correction adapts to vintage tones.

---

## 🔬 Technical Deep Dive

### Color Correction Algorithm

ReActor V3 uses **LAB color space transfer**:

1. **Extract face region** from both swapped and target images
2. **Convert BGR → LAB** (perceptually uniform color space)
3. **Calculate statistics:**
   - Mean (L*, a*, b*) for lighting and color
   - Std deviation for contrast
4. **Transfer:**
   ```
   swapped_lab = (swapped_lab - swapped_mean) × (target_std / swapped_std) + target_mean
   ```
5. **Convert LAB → BGR** and paste back

**Result:** Swapped face inherits target's lighting, color tone, and contrast.

---

### Blend Strength Implementation

Uses OpenCV's **alpha blending**:

```python
result = cv2.addWeighted(swapped, blend_ratio, original, 1 - blend_ratio, 0)
```

**Example:**
- Blend = 1.0: 100% swapped, 0% original
- Blend = 0.8: 80% swapped, 20% original (subtle original features)
- Blend = 0.5: 50/50 hybrid

---

### Upscale Factor Mechanism

Affects **FaceRestoreHelper's upscale_factor**:

1. Face detection finds face bbox
2. **Upscale factor determines extraction size:**
   - 1x: Extract face at detected size
   - 2x: Extract face at 2× detected size
3. GPEN processes the larger face
4. Result is pasted back at original scale

**Technical:**
- Detection: 256×256 face → Extract 512×512 → GPEN restores → Paste 256×256
- Gain: Higher input resolution = sharper details after restoration

---

## 🚀 Performance Optimization

### VRAM Usage by Configuration

| Config | GPEN | Upscale | VRAM | Speed |
|--------|------|---------|------|-------|
| Minimal | 512 | 1x | 4GB | ~180ms |
| Balanced | 512 | 1x | 6GB | ~200ms |
| High Quality | 1024 | 1x | 8GB | ~350ms |
| Ultra Quality | 1024 | 2x | 12GB | ~600ms |

*Tested on RTX 3060*

---

### Speed Tips

1. **Use 512 for batch processing** - 2.5× faster than 1024
2. **Adjust resolution threshold** - Higher = more 512 usage = faster
3. **Disable upscale (1x)** - 2× faster vs 2x upscale
4. **Lower detection threshold** - Faster face detection
5. **Enable aggressive cleanup** - Consistent speed across batches

---

## 🎓 Best Practices

### ✅ DO:
- **Always enable color correction** for realism
- **Use Smart Match (S)** to avoid gender mismatches
- **Start with defaults** (blend=1.0, detection=0.5, upscale=1x)
- **Use GPEN-1024 for portraits**, GPEN-512 for groups
- **Enable aggressive cleanup** if <12GB VRAM
- **Use well-lit, frontal source faces** for best results

### ❌ DON'T:
- Don't use blend < 0.7 expecting full swaps (it's a hybrid)
- Don't set detection > 0.8 (too strict, misses faces)
- Don't use 2x upscale without 12GB+ VRAM
- Don't disable color correction unless intentional
- Don't use extreme angles or occluded faces (detection will fail)

---

## 🐛 Troubleshooting

### "No face detected" despite visible face
- **Lower detection threshold** to 0.3-0.4
- Ensure face is >64px and not heavily occluded
- Try frontal or near-frontal pose

### Swapped face has wrong colors
- **Enable color correction**
- Check source image has neutral lighting

### Results are blurry
- **Use GPEN-1024** instead of 512
- **Enable 2x upscale** (requires more VRAM)
- Lower resolution threshold (256-320)

### Out of VRAM errors
- **Enable aggressive cleanup**
- Use GPEN-512 instead of 1024
- Disable upscale (1x)
- Close other applications

### Face looks "pasted on" / unrealistic
- **Enable color correction**
- Lower blend to 0.92-0.97
- Ensure good lighting match between source/target

---

## 📊 Quality Comparison Matrix

| Setting | Detail Level | Realism | Speed | VRAM |
|---------|--------------|---------|-------|------|
| **512 + 1x** | High | ⭐⭐⭐⭐ | Fast | Low |
| **512 + 2x** | Very High | ⭐⭐⭐⭐⭐ | Medium | Medium |
| **1024 + 1x** | Very High | ⭐⭐⭐⭐⭐ | Medium | Medium |
| **1024 + 2x** | Ultra | ⭐⭐⭐⭐⭐ | Slow | High |

---

## 🎯 Summary

**For Maximum Realism:**
1. Use **GPEN-1024** with **2x upscale** (if VRAM allows)
2. Enable **color correction** (always)
3. Use **blend = 0.92-0.98** (slight blend is more natural than 1.0)
4. Use **Smart Match (S)** gender mode
5. Set **detection threshold = 0.5-0.6** (quality control)
6. Lower **resolution threshold to 256-320px** (aggressive 1024 usage)

**Result:** Photorealistic face swaps with:
- Skin pores and micro-textures visible
- Iris patterns and eyelash detail
- Perfect lighting/color match
- Natural blending at edges
- Indistinguishable from real photos

---

*ReActor V3 - Professional Face Swapping for Stable Diffusion WebUI Forge*

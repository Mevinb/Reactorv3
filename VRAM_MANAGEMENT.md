# ReActor V3 - VRAM Management Guide

## Problem Summary

When using high-power models like Juggernaut XL 9 with ReActor V3, you may experience:
- **First generation works fine** - LoRA and Juggernaut use VRAM normally
- **ReActor V3 loads** - Takes up significant VRAM for face swapping/restoration
- **Models don't unload** - ReActor V3 stays in memory instead of releasing VRAM
- **Subsequent generations slow down** - Less available VRAM causes performance degradation

## Solution: Automatic VRAM Cleanup

ReActor V3 now includes **automatic memory management** that frees VRAM after each generation.

### How It Works

1. **Automatic Cleanup (Default)**
   - After each face swap operation, ReActor V3 automatically clears CUDA cache
   - Uses WebUI Forge's built-in memory management system
   - Keeps models cached for performance (recommended)

2. **Aggressive Cleanup (Optional)**
   - Unloads ALL cached models including GPEN restoration models
   - Frees maximum VRAM but slower for batch processing
   - Recommended for systems with <12GB VRAM

## Usage

### In WebUI Interface

When using ReActor V3 in the WebUI:

1. Open the **"ReActor V3 - GPEN High-Fidelity Face Swap"** accordion
2. Enable the extension with the checkbox
3. **For limited VRAM**: Check **"Aggressive Memory Cleanup"**

```
☐ Enable ReActor V3
☑ Auto-Select Resolution
☑ Aggressive Memory Cleanup    ← Enable this for <12GB VRAM
```

### Manual Cleanup Script

If you experience memory issues, run the manual cleanup utility:

```bash
# From the extension directory
cd webui/extensions/sd-webui-reactor-v3
python clear_vram.py
```

This will:
- Display current VRAM usage
- Clear all model caches
- Force CUDA cache cleanup
- Run Python garbage collection
- Show freed memory

## VRAM Usage Comparison

### Before Fix
```
Generation 1: 10GB VRAM → Fast ✓
ReActor loads: +4GB VRAM
Generation 2: 14GB VRAM → Slow ✗ (Out of memory, using system RAM)
Generation 3: 14GB VRAM → Slow ✗
```

### After Fix (Automatic Cleanup)
```
Generation 1: 10GB VRAM → Fast ✓
ReActor loads: +4GB VRAM
Cleanup: -2GB VRAM (cache cleared)
Generation 2: 12GB VRAM → Fast ✓
ReActor reuses cache: +2GB VRAM
Cleanup: -2GB VRAM
Generation 3: 12GB VRAM → Fast ✓
```

### After Fix (Aggressive Cleanup)
```
Generation 1: 10GB VRAM → Fast ✓
ReActor loads: +4GB VRAM
Aggressive cleanup: -4GB VRAM (models unloaded)
Generation 2: 10GB VRAM → Fast ✓ (but ReActor reloads models)
ReActor reloads: +4GB VRAM
Aggressive cleanup: -4GB VRAM
Generation 3: 10GB VRAM → Fast ✓
```

## Configuration Options

### Auto Cleanup (Default: Enabled)

Located in `reactor_v3_swapper.py`:
```python
self.auto_cleanup = True  # Set to False to disable automatic cleanup
```

### Cleanup Modes

**Normal Cleanup** (`aggressive=False`):
- Clears CUDA cache
- Runs garbage collection
- Keeps InsightFace and GPEN models loaded for faster reuse
- **Use when**: VRAM >12GB, batch processing

**Aggressive Cleanup** (`aggressive=True`):
- Everything in Normal mode +
- **Unloads InsightFace models** (face analyzer ~1GB, face swapper ~500MB)
- Unloads all GPEN restoration models
- Forces complete model reload on next use (adds 3-5s loading time)
- **Use when**: VRAM <12GB, memory constrained, or using inswapper_128

## Technical Details

### Memory Management Functions

The solution integrates with WebUI Forge's memory management:

1. **`soft_empty_cache()`** - Clears PyTorch CUDA cache
2. **`unload_all_models()`** - Releases all loaded models
3. **`gc.collect()`** - Python garbage collection
4. **`clear_gpen_cache()`** - Clears ReActor V3 model cache

### Code Changes

**Added to `reactor_v3_swapper.py`**:
```python
def cleanup_memory(self, aggressive: bool = False):
    """Free VRAM after processing"""
    if aggressive:
        clear_gpen_cache()
    memory_management.soft_empty_cache(force=True)
    gc.collect()
```

**Automatic cleanup after each process**:
```python
if self.auto_cleanup:
    self.cleanup_memory(aggressive=False)
```

## Troubleshooting

### Still experiencing slowdowns?

1. **Enable Aggressive Cleanup** in the UI
2. **Run manual cleanup**:
   ```bash
   python clear_vram.py
   ```
3. **Use lower resolution models**:
   - GPEN-512 instead of GPEN-1024
   - Saves ~2GB VRAM

4. **Check VRAM usage**:
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
   ```

### Models not loading?

If cleanup is too aggressive and models won't load:
```python
# In reactor_v3_swapper.py, line ~75
self.auto_cleanup = False  # Disable automatic cleanup
```

## Performance Recommendations

| VRAM | Recommended Settings |
|------|---------------------|
| 8GB | Aggressive Cleanup ON, GPEN-512 only |
| 10-12GB | Aggressive Cleanup ON, GPEN-512/1024 |
| 16GB+ | Normal Cleanup (default), Any model |
| 24GB+ | Can disable cleanup entirely |

## Monitoring

The extension prints VRAM stats after cleanup:
```
[ReActor V3] Cleaning up memory (aggressive=False)...
[ReActor V3] Used Forge memory management for cleanup
[ReActor V3] VRAM - Allocated: 8.45 GB, Reserved: 9.12 GB
```

Watch your console for these messages to verify cleanup is working.

## Additional Tips

1. **Restart WebUI** if memory issues persist
2. **Close other GPU applications** (browsers with hardware acceleration, etc.)
3. **Use `--medvram` or `--lowvram`** launch flags if needed
4. **Monitor with nvidia-smi**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Credits

This VRAM management solution integrates with:
- WebUI Forge's memory_management system
- PyTorch CUDA memory management
- InsightFace model caching
- ONNX Runtime session handling

---

**Last Updated**: 2026-01-02
**Version**: ReActor V3 with Automatic VRAM Management

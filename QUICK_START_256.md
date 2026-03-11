# ReActor V3 - 256 Swapper Quick Start Guide

## 🎯 What's New?

ReActor V3 now supports **multiple face swapper models** with automatic resolution detection:

- **inswapper_128.onnx** — Legacy model (128x128)
- **reswapper_256.onnx** — Best for identity preservation (256x256) ⭐ 
- **hyperswap_1a_256.onnx** — Best for sharpness (256x256)

Everything automatically adapts: feathering, GPEN strength, identity protection, and texture processing.

---

## 🚀 Quick Start

### Step 1: Install Models

Download swapper models and place them in:
```
extensions/sd-webui-reactor-v3/models/insightface/
```

### Step 2: Select Model in UI

1. Open **ReActor V3** accordion in img2img or txt2img
2. Find the **"Face Swapper Model"** dropdown (new!)
3. Select your desired model:
   - `inswapper_128.onnx` — Standard/Legacy
   - `reswapper_256.onnx` — Best identity ⭐
   - `hyperswap_1a_256.onnx` — Sharpest edges

### Step 3: Generate!

That's it! The system automatically:
- Detects the model resolution (128 or 256)
- Adjusts feathering for sharper edges
- Caps GPEN alpha to prevent over-processing
- Applies stricter identity protection
- Reduces extra sharpening/texture

---

## 🎨 Which Model Should I Use?

### Use **reswapper_256.onnx** when:
- Identity accuracy is critical
- You want natural skin tones
- Face-to-face swaps (portraits)
- Minimal seam artifacts

### Use **hyperswap_1a_256.onnx** when:
- Maximum sharpness is needed
- High-detail close-ups
- Texture preservation is important

### Use **inswapper_128.onnx** when:
- Legacy compatibility needed
- Limited VRAM (<8GB)
- Quick tests/previews

---

## ⚙️ Auto-Select Mode

Enable **"Auto-Select Swapper"** checkbox to let the system choose the best model for you.

Current behavior:
- Defaults to `reswapper_256.onnx` for best quality

---

## 🔍 What Happens Behind the Scenes?

### For 256 Models:
- **Feathering:** 5% diagonal (sharper edges)
- **GPEN Alpha:** Capped at 0.35 (less enhancement)
- **Identity Threshold:** 0.85 (stricter)
- **Texture Injection:** Reduced (prevents white streaks)
- **Extra Sharpening:** Limited (avoids artifacts)

### For 128 Models (Legacy):
- **Feathering:** 8% diagonal (more blending)
- **GPEN Alpha:** Capped at 0.45 (standard)
- **Identity Threshold:** 0.60 (standard)
- **Texture Injection:** Normal
- **Extra Sharpening:** Normal

All of this happens **automatically** based on the detected resolution!

---

## 📋 Console Logs

You'll see these new messages:

```
[ReActor V3] ── Initializing Face Swapper ──
[ReActor V3]   Swapper detected: reswapper_256.onnx (256x256)
[ReActor V3]   Resolution-aware blending enabled
[ReActor V3] GPEN alpha cap for 256 swapper: 0.42 → 0.35
[ReActor V3] GPEN identity guardrail: threshold=0.85
[ReActor V3] Face Fixer: 256 swapper detected — reducing extra sharpening/texture
```

---

## ⚡ Performance

### VRAM Usage:
- **128 model:** ~500-600 MB
- **256 model:** ~800-1000 MB

### Speed:
- **256 models:** ~10-20% slower than 128
- Still very fast! Most time is spent in GPEN restoration

### Quality:
- **256 models:** Significantly better identity & edges
- **Less GPEN needed:** Native detail is already good

---

## 🎓 Tips & Tricks

1. **Start with reswapper_256** — It's the best all-rounder
2. **Use Aggressive Cleanup** — Enable if you have <12GB VRAM
3. **GPEN-512 recommended** — Pairs well with 256 swappers
4. **Identity guardrail** — If you see "identity drift detected", alpha is auto-reduced
5. **Backward compatible** — inswapper_128 works exactly as before

---

## 🐛 Troubleshooting

### "Model not found"
- Check model is in `extensions/sd-webui-reactor-v3/models/insightface/`
- Verify filename matches exactly (case-sensitive)

### "Using default 128"
- Model detection failed, falling back to 128
- This is safe — functionality still works

### "Out of VRAM"
- Switch to inswapper_128 (smaller)
- Enable "Aggressive Memory Cleanup"
- Reduce GPEN resolution (512 → None)

### Results look plastic/over-processed
- This shouldn't happen with 256 models!
- Check console — identity guardrail should trigger
- Try lowering GPEN fidelity weight if available

### Seams too visible
- 128 models need more feathering (expected)
- Switch to 256 model for tighter seams
- Enable "Auto Face Detail Fix"

---

## 📚 Advanced: Model Switching in Code

```python
# Get engine instance
engine = get_reactor_v3_engine(models_path)

# Switch to 256 model
engine.reload_face_swapper("reswapper_256.onnx")

# Check current resolution
print(f"Current swapper: {engine.swapper_name}")
print(f"Resolution: {engine.swapper_input_size}x{engine.swapper_input_size}")

# Process with the new model
result, msg = engine.process_auto_match(source, target, restore_model="GPEN-BFR-512.onnx")
```

---

## 🎯 Key Takeaways

✅ **No configuration needed** — Everything auto-detects  
✅ **Fully backward compatible** — inswapper_128 still default  
✅ **Better quality** — 256 models = better identity & edges  
✅ **Smarter processing** — Less GPEN/sharpen for clean results  
✅ **Memory efficient** — Automatic cleanup when switching  

---

**Enjoy the upgrade! 🎉**

For detailed technical documentation, see: `REACTOR_V3_256_UPGRADE.md`

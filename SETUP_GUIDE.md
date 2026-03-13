# ReActor V3 - Quick Setup Guide

This is a streamlined setup guide. For full documentation, see [README.md](README.md).

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install the Extension

```bash
cd stable-diffusion-webui-forge/extensions
git clone https://github.com/yourusername/sd-webui-reactor-v3.git
```

### Step 2: First-Swap Auto Model Download

On your first actual swap attempt, ReActor V3 auto-downloads required models:

- `GPEN-BFR-512.onnx` to `models/facerestore_models/`
- `inswapper_128.onnx` to `models/insightface/`

No manual model copy is required for these defaults.

**Optional:** Download `GPEN-BFR-1024.onnx` manually for ultra quality and place it in `models/facerestore_models/`.

### Step 3: Restart WebUI

Restart Stable Diffusion WebUI Forge. Dependencies will auto-install.

### Step 4: Use It!

1. Open the **"ReActor V3"** tab
2. Upload source face image
3. Upload target image
4. Select GPEN model from dropdown
5. Click **⚡ Swap Face**

**Done!** 🎉

---

## 📁 Directory Structure

After setup, your directory should look like:

```
extensions/sd-webui-reactor-v3/
├── scripts/                        ← Python code
├── install.py
├── requirements.txt
└── README.md
```

Shared WebUI model folders used at runtime:

```
webui/models/
├── facerestore_models/
│   ├── GPEN-BFR-512.onnx           ← Auto-downloaded on first swap
│   └── GPEN-BFR-1024.onnx          ← Optional manual add
└── insightface/
   └── inswapper_128.onnx          ← Auto-downloaded on first swap
```

---

## ⚠️ Common First-Time Issues

### Issue 1: "No face detected"
✅ **Fix:** Ensure face is clearly visible, well-lit, and frontal

### Issue 2: "Model not found"  
✅ **Fix:** Trigger one swap so first-run auto-download runs, then verify `models/facerestore_models/`

### Issue 3: First run is slow
✅ **This is normal:** ReActor V3 is downloading required models on first swap (GPEN-512 + inswapper), and InsightFace may also download buffalo_l (~500MB)

### Issue 4: Import errors
✅ **Fix:** Manually install dependencies:
```bash
pip install insightface onnxruntime-gpu opencv-python
```

---

## 🎯 Model Selection Guide

| Your Goal | Use This Model |
|-----------|---------------|
| Fast results, good quality | GPEN-512 |
| Maximum quality, close-ups | GPEN-1024 |
| Not sure? | GPEN-512 + Auto-Resolution ✓ |

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 6GB | RTX 3060 12GB+ |
| VRAM | 4GB (512 model) | 8GB (1024 model) |
| RAM | 8GB | 16GB |
| CUDA | 11.x or 12.x | 12.x |
| OS | Windows 10+ / Linux | Any |

---

## 📊 Quick Performance Reference

**NVIDIA RTX 3060 12GB:**
- GPEN-512: ~230ms total (fast)
- GPEN-1024: ~400ms total (medium)

**NVIDIA RTX 4090:**
- GPEN-512: ~120ms total (very fast)
- GPEN-1024: ~200ms total (fast)

---

## 🔗 Quick Links

- **Full Documentation:** [README.md](README.md)
- **Troubleshooting:** See README.md → Troubleshooting section
- **GPEN-512 Download:** https://huggingface.co/yangxy/GPEN
- **GPEN-1024 Download:** https://modelscope.cn/models/damo/GPEN-BFR-1024
- **Report Issues:** Open an issue on GitHub

---

## ✅ Verification Checklist

Before asking for help, verify:

- [ ] Extension folder exists: `extensions/sd-webui-reactor-v3/`
- [ ] Ran one swap once to trigger auto-download
- [ ] GPEN model exists: `models/facerestore_models/GPEN-BFR-512.onnx`
- [ ] InSwapper exists: `models/insightface/inswapper_128.onnx`
- [ ] WebUI restarted after installation
- [ ] "ReActor V3" tab appears in WebUI
- [ ] Console shows no red error messages
- [ ] Dependencies installed: `pip list | grep insightface`

---

## 🆘 Still Need Help?

1. Check console output for error messages
2. Read full documentation: [README.md](README.md)
3. Open an issue with:
   - Console logs
   - GPU model and VRAM
   - WebUI Forge version
   - Steps to reproduce

---

<div align="center">

**Happy Face Swapping!** 🎨

*For advanced usage, technical details, and API integration, see [README.md](README.md)*

</div>

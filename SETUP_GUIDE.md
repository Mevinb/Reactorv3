# ReActor V3 - Quick Setup Guide

This is a streamlined setup guide. For full documentation, see [README.md](README.md).

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install the Extension

```bash
cd stable-diffusion-webui-forge/extensions
git clone https://github.com/yourusername/sd-webui-reactor-v3.git
```

### Step 2: Download GPEN Models

Download at least one GPEN model:

**Option A: GPEN-512 (Recommended - Faster)**
- Download: [GPEN-BFR-512.onnx](https://huggingface.co/yangxy/GPEN/blob/main/GPEN-BFR-512.onnx)
- Size: ~330 MB
- Place in: `extensions/sd-webui-reactor-v3/models/facerestore_models/`

**Option B: GPEN-1024 (Ultra Quality - Slower)**
- Download: [GPEN-BFR-1024.onnx](https://modelscope.cn/models/damo/GPEN-BFR-1024)
- Size: ~330 MB
- Place in: `extensions/sd-webui-reactor-v3/models/facerestore_models/`

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
├── models/
│   ├── facerestore_models/
│   │   ├── GPEN-BFR-512.onnx      ← Put GPEN models here
│   │   └── GPEN-BFR-1024.onnx     ← Optional but recommended
│   └── insightface/                ← Auto-downloaded (first run)
├── scripts/                        ← Python code
├── install.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Common First-Time Issues

### Issue 1: "No face detected"
✅ **Fix:** Ensure face is clearly visible, well-lit, and frontal

### Issue 2: "Model not found"  
✅ **Fix:** Verify GPEN .onnx files are in `models/facerestore_models/`

### Issue 3: First run is slow
✅ **This is normal:** InsightFace is downloading ~500MB of face detection models

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
- [ ] GPEN model exists: `models/facerestore_models/GPEN-BFR-512.onnx`
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

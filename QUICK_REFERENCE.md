# 🎭 Gender Matching - Quick Reference

## 4 Modes Available

### 🌟 Smart Match (S) - RECOMMENDED
**What it does:** Auto-detects source gender → Only swaps matching gender in target

**Use when:**
- Target has multiple people of different genders
- You want automatic gender-based selection
- General everyday use

**Example:**
- Source: Female portrait
- Target: Wedding photo (bride + groom)
- **Result:** Only bride's face is swapped ✓

---

### 🚫 All (A) - No Filter
**What it does:** Traditional face swap, ignores gender

**Use when:**
- Gender doesn't matter
- Creative/artistic effects
- Quick swaps without filtering

**Example:**
- Source: Any face
- Target: Any face
- **Result:** First detected face is swapped

---

### 👨 Male Only (M)
**What it does:** ONLY processes male faces in both images

**Use when:**
- Both images contain only males
- You want to ensure male-to-male swap
- Batch processing male faces

**Example:**
- Source: Male actor
- Target: Group of men
- **Result:** Only male faces considered

---

### 👩 Female Only (F)
**What it does:** ONLY processes female faces in both images

**Use when:**
- Both images contain only females
- You want to ensure female-to-female swap
- Batch processing female faces

**Example:**
- Source: Female model
- Target: Group of women
- **Result:** Only female faces considered

---

## How to Use

1. **Open ReActor V3 section** in img2img or txt2img
2. **Upload source face** (the face you want to copy)
3. **Enable ReActor V3** checkbox
4. **Select Gender Matching Mode:**
   - For most cases: **Smart Match (S)**
   - For anything goes: **All (A)**
   - For specific gender: **Male (M)** or **Female (F)**
5. **Generate your image**

---

## Console Messages

### Smart Match Success
```
[ReActor V3] Smart Match: Source gender = F
[ReActor V3] Face 0: Gender=M ✗
[ReActor V3] Face 1: Gender=F ✓
[ReActor V3] Found 1 matching face(s)
```

### No Match Found
```
Error: No female face found in target to match source
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No matching face found" | Switch to "All (A)" mode |
| Wrong face selected | Adjust "Target Face Index" slider |
| Gender detected incorrectly | Use "All (A)" to bypass detection |

---

## Pro Tips

💡 **Mixed Group Photos:** Always use Smart Match (S)

💡 **Same Gender Groups:** Use Male (M) or Female (F) for consistency

💡 **Artistic Freedom:** Use All (A) for creative effects

💡 **Check Console:** Watch for gender detection results (M/F/U)

💡 **Face Index:** Index refers to filtered faces, not all faces

---

## What Each Mode Returns

| Mode | Source | Target | Result |
|------|--------|--------|--------|
| **S** (Smart) | Female | 2 men + 1 woman | Swaps the woman only |
| **S** (Smart) | Male | 2 women + 1 man | Swaps the man only |
| **M** (Male) | Male | Mixed group | Swaps male faces only |
| **F** (Female) | Female | Mixed group | Swaps female faces only |
| **A** (All) | Any | Any | Swaps first face (no filter) |

---

## Quick Decision Guide

```
Do you have mixed genders in target?
├─ YES → Use Smart Match (S) ⭐
└─ NO → Do you care about gender?
    ├─ YES → Use Male (M) or Female (F)
    └─ NO → Use All (A)
```

---

## Default Settings

✅ Gender Matching: **Smart Match (S)**  
✅ Auto Resolution: **Enabled**  
✅ Aggressive Cleanup: **Enabled**  

These are good defaults for most users!

---

**Need more details?** See `GENDER_MATCHING_GUIDE.md`

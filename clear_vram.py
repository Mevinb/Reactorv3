"""
ReActor V3 - Manual VRAM Cleanup Utility

This script can be run independently to force cleanup of all cached models
and free up VRAM. Useful for troubleshooting or when you notice memory issues.

Usage:
    python clear_vram.py
"""

import os
import sys
import gc
import torch

# Add WebUI to path to access memory management
script_dir = os.path.dirname(os.path.abspath(__file__))
extension_dir = os.path.dirname(script_dir)
extensions_dir = os.path.dirname(extension_dir)
webui_dir = os.path.dirname(extensions_dir)
sys.path.insert(0, webui_dir)

try:
    from backend import memory_management
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False

# Import the cleanup functions
from scripts.reactor_v3_gpen_restorer_new import clear_gpen_cache


def check_vram_status():
    """Display current VRAM usage"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return
    
    print("\n" + "="*60)
    print("VRAM Status Before Cleanup")
    print("="*60)
    
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(device)
        
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        total = props.total_memory / (1024**3)
        
        print(f"GPU {i}: {props.name}")
        print(f"  Total VRAM:     {total:.2f} GB")
        print(f"  Allocated:      {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved:       {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Free:           {total - allocated:.2f} GB")


def perform_cleanup():
    """Perform aggressive VRAM cleanup"""
    print("\n" + "="*60)
    print("Performing Aggressive Cleanup")
    print("="*60)
    
    # Step 1: Clear GPEN cache
    print("1. Clearing GPEN model cache...")
    clear_gpen_cache()
    
    # Step 2: Use WebUI memory management if available
    if MEMORY_MANAGEMENT_AVAILABLE:
        print("2. Using WebUI Forge memory management...")
        memory_management.soft_empty_cache(force=True)
        
        # Also try to unload all models
        try:
            memory_management.unload_all_models()
            print("   ✓ Unloaded all models")
        except Exception as e:
            print(f"   ⚠ Could not unload all models: {e}")
    else:
        print("2. WebUI memory management not available")
    
    # Step 3: Manual CUDA cleanup
    if torch.cuda.is_available():
        print("3. Performing manual CUDA cleanup...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("   ✓ CUDA cache cleared")
    else:
        print("3. CUDA not available")
    
    # Step 4: Python garbage collection
    print("4. Running Python garbage collection...")
    collected = gc.collect()
    print(f"   ✓ Collected {collected} objects")
    
    print("\n✅ Cleanup complete!")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("ReActor V3 - VRAM Cleanup Utility")
    print("="*60)
    
    # Check initial status
    check_vram_status()
    
    # Perform cleanup
    perform_cleanup()
    
    # Check status after cleanup
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("VRAM Status After Cleanup")
        print("="*60)
        
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            total = props.total_memory / (1024**3)
            freed = reserved - allocated
            
            print(f"GPU {i}: {props.name}")
            print(f"  Allocated:      {allocated:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"  Reserved:       {reserved:.2f} GB ({reserved/total*100:.1f}%)")
            print(f"  Free:           {total - allocated:.2f} GB")
            if freed > 0:
                print(f"  ✓ Freed:        {freed:.2f} GB")
    
    print("\n" + "="*60)
    print("💡 Tips:")
    print("  - Run this script when you notice slowdowns")
    print("  - Enable 'Aggressive Memory Cleanup' in WebUI for automatic cleanup")
    print("  - Consider using lower resolution models (512 vs 1024) if VRAM limited")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

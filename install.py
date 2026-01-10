"""
ReActor V3 Installation Script

This script runs during extension installation to ensure all dependencies are met.
"""

import launch
import os


def install_requirements():
    """Install required packages for ReActor V3"""
    
    print("[ReActor V3] Checking dependencies...")
    
    # Core dependencies
    requirements = [
        'insightface',
        'onnxruntime-gpu',  # GPU acceleration
        'opencv-python',
    ]
    
    for package in requirements:
        try:
            if not launch.is_installed(package):
                print(f"[ReActor V3] Installing {package}...")
                launch.run_pip(f"install {package}", f"ReActor V3 requirement: {package}")
            else:
                print(f"[ReActor V3] {package} already installed")
        except Exception as e:
            print(f"[ReActor V3] WARNING: Failed to install {package}: {e}")
            print(f"[ReActor V3] You may need to install manually: pip install {package}")
    
    print("[ReActor V3] Dependency check complete")


# Run installation
install_requirements()

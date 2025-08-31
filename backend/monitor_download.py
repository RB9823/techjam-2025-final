#!/usr/bin/env python3
"""
Monitor model download progress
"""
import time
import os
from pathlib import Path

def monitor_cache():
    """Monitor HuggingFace cache growth"""
    cache_dir = Path.home() / ".cache" / "huggingface" 
    
    print("📊 Monitoring model download progress...")
    print("Press Ctrl+C to stop monitoring")
    
    initial_size = 0
    if cache_dir.exists():
        initial_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    
    print(f"Initial cache size: {initial_size / (1024**3):.2f} GB")
    
    try:
        while True:
            if cache_dir.exists():
                current_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                growth = current_size - initial_size
                
                print(f"\r📥 Downloaded: {growth / (1024**3):.2f} GB (Total: {current_size / (1024**3):.2f} GB)", end="", flush=True)
                
                # Check for model directories
                clip_dir = cache_dir / "models--openai--clip-vit-large-patch14"
                florence_dir = cache_dir / "models--microsoft--Florence-2-base"
                omni_dir = cache_dir / "models--microsoft--OmniParser-v2.0"
                
                if clip_dir.exists():
                    print(f"\n   ✅ CLIP model directory found")
                if florence_dir.exists():
                    print(f"\n   ✅ Florence-2 model directory found")
                if omni_dir.exists():
                    print(f"\n   ✅ OmniParser model directory found")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n\n📊 Final cache size: {current_size / (1024**3):.2f} GB")
        print("Monitor stopped.")

if __name__ == "__main__":
    monitor_cache()
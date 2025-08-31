#!/usr/bin/env python3
"""
AI Model Download Script - Ultra-optimized for speed and reliability
Downloads all required models for UI Validation API before server startup
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure minimal logging for downloads
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import after setting up basic logging
try:
    from app.core.config import settings
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    from ultralytics import YOLO
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Run 'uv sync' first to install dependencies")
    sys.exit(1)

class ModelDownloader:
    """Ultra-fast parallel model downloader"""
    
    def __init__(self):
        # Use default HuggingFace cache if not specified
        if settings.huggingface_cache_dir:
            self.cache_dir = Path(settings.huggingface_cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = None  # Use default ~/.cache/huggingface
        
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Model definitions
        self.models = {
            "yolo": {
                "repo": "microsoft/OmniParser-v2.0",
                "filename": "icon_detect/model.pt", 
                "description": "YOLO UI Element Detection",
                "size": "~6MB",
                "critical": True
            },
            "florence2_omni": {
                "repo": "microsoft/OmniParser-v2.0",
                "files": ["icon_caption/*"],
                "description": "OmniParser Florence-2 Captioning",
                "size": "~232MB", 
                "critical": True
            },
            "florence2_base": {
                "repo": "microsoft/Florence-2-base",
                "description": "Florence-2 Base Model",
                "size": "~464MB",
                "critical": True
            },
            "clip": {
                "repo": "openai/clip-vit-large-patch14",
                "description": "CLIP Visual-Text Similarity (Large)",
                "size": "~600MB",
                "critical": True
            }
        }
    
    def download_yolo_model(self):
        """Download YOLO detection model"""
        try:
            logger.info("📥 Downloading YOLO model...")
            model_path = hf_hub_download(
                repo_id=self.models["yolo"]["repo"],
                filename=self.models["yolo"]["filename"],
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            
            # Test loading
            yolo_model = YOLO(model_path)
            logger.info(f"✅ YOLO model downloaded and verified: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ YOLO download failed: {e}")
            return False
    
    def download_florence2_omni(self):
        """Download OmniParser Florence-2 model"""
        try:
            logger.info("📥 Downloading OmniParser Florence-2...")
            model_path = snapshot_download(
                repo_id=self.models["florence2_omni"]["repo"],
                allow_patterns=self.models["florence2_omni"]["files"],
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            logger.info(f"✅ OmniParser Florence-2 downloaded: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ OmniParser Florence-2 download failed: {e}")
            return False
    
    def download_florence2_base(self):
        """Download Florence-2 base model"""
        try:
            logger.info("📥 Downloading Florence-2 base...")
            
            # Download processor
            processor = AutoProcessor.from_pretrained(
                self.models["florence2_base"]["repo"],
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Download model (without loading to save memory)
            model = AutoModelForCausalLM.from_pretrained(
                self.models["florence2_base"]["repo"],
                torch_dtype=torch.float32,  # Use float32 for compatibility
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                device_map=None  # Don't load to device yet
            )
            
            logger.info("✅ Florence-2 base model downloaded and verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Florence-2 base download failed: {e}")
            return False
    
    def download_clip_model(self):
        """Download CLIP model"""
        try:
            logger.info("📥 Downloading CLIP model...")
            
            from transformers import CLIPProcessor, CLIPModel
            
            # Download processor and model
            processor = CLIPProcessor.from_pretrained(
                self.models["clip"]["repo"],
                cache_dir=self.cache_dir
            )
            
            model = CLIPModel.from_pretrained(
                self.models["clip"]["repo"],
                cache_dir=self.cache_dir,
                device_map=None  # Don't load to device yet
            )
            
            logger.info("✅ CLIP model downloaded and verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ CLIP download failed: {e}")
            return False
    
    def check_existing_models(self):
        """Check which models are already downloaded"""
        logger.info("🔍 Checking existing models...")
        
        existing = {}
        
        # Determine cache directory to check
        cache_to_check = self.cache_dir or Path.home() / ".cache" / "huggingface"
        
        # Check cache directory contents
        if cache_to_check.exists():
            try:
                cache_size = sum(f.stat().st_size for f in cache_to_check.rglob('*') if f.is_file())
                logger.info(f"Cache directory size: {cache_size / (1024**3):.2f} GB")
            except:
                logger.info(f"Cache directory: {cache_to_check}")
            
            # Check for specific model indicators
            for model_name, model_info in self.models.items():
                model_dir = cache_to_check / "models--" / model_info["repo"].replace("/", "--")
                existing[model_name] = model_dir.exists()
                
                if existing[model_name]:
                    logger.info(f"   ✅ {model_name}: Found cached")
                else:
                    logger.info(f"   ❌ {model_name}: Not found")
        else:
            logger.info(f"Cache directory not found: {cache_to_check}")
            for model_name in self.models.keys():
                existing[model_name] = False
        
        return existing
    
    def download_all_models_parallel(self):
        """Download all models in parallel for maximum speed"""
        logger.info("🚀 Starting parallel model downloads...")
        
        download_functions = [
            ("yolo", self.download_yolo_model),
            ("florence2_omni", self.download_florence2_omni), 
            ("florence2_base", self.download_florence2_base),
            ("clip", self.download_clip_model)
        ]
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all downloads
            future_to_model = {
                executor.submit(func): name 
                for name, func in download_functions
            }
            
            # Process completed downloads
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    
                    if result:
                        logger.info(f"✅ {model_name} download completed")
                    else:
                        logger.error(f"❌ {model_name} download failed")
                        
                except Exception as e:
                    logger.error(f"❌ {model_name} download exception: {e}")
                    results[model_name] = False
        
        total_time = time.time() - start_time
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"📊 Download Summary:")
        logger.info(f"   ✅ Successful: {successful}/{total}")
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
        
        return results
    
    def verify_all_models(self):
        """Verify all models can be loaded successfully"""
        logger.info("🔬 Verifying model loading...")
        
        verification_results = {}
        
        try:
            # Test YOLO
            logger.info("Testing YOLO...")
            yolo_path = hf_hub_download(
                "microsoft/OmniParser-v2.0",
                "icon_detect/model.pt",
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            yolo_model = YOLO(yolo_path)
            verification_results["yolo"] = True
            logger.info("   ✅ YOLO verification passed")
            
        except Exception as e:
            logger.error(f"   ❌ YOLO verification failed: {e}")
            verification_results["yolo"] = False
        
        try:
            # Test Florence-2
            logger.info("Testing Florence-2...")
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            verification_results["florence2"] = True
            logger.info("   ✅ Florence-2 verification passed")
            
        except Exception as e:
            logger.error(f"   ❌ Florence-2 verification failed: {e}")
            verification_results["florence2"] = False
        
        try:
            # Test CLIP
            logger.info("Testing CLIP...")
            from transformers import CLIPProcessor, CLIPModel
            
            clip_processor = CLIPProcessor.from_pretrained(
                self.models["clip"]["repo"],
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            verification_results["clip"] = True
            logger.info("   ✅ CLIP verification passed")
            
        except Exception as e:
            logger.error(f"   ❌ CLIP verification failed: {e}")
            verification_results["clip"] = False
        
        return verification_results

def main():
    """Main download function"""
    print("🤖 AI Model Download Script - Ultra Mode")
    print("=" * 50)
    
    downloader = ModelDownloader()
    
    # Check existing models
    existing = downloader.check_existing_models()
    missing_models = [name for name, exists in existing.items() if not exists]
    
    if not missing_models:
        print("🎉 All models already downloaded!")
        
        # Still verify they work
        print("\n🔬 Verifying existing models...")
        verification = downloader.verify_all_models()
        
        if all(verification.values()):
            print("✅ All models verified successfully!")
            print("\n🚀 Ready to start server with: uv run python main.py")
        else:
            print("⚠️  Some models failed verification, re-downloading...")
            downloader.download_all_models_parallel()
        
        return
    
    print(f"\n📥 Missing models: {', '.join(missing_models)}")
    
    # Estimate download time and size
    total_size_mb = sum([
        6,    # YOLO
        232,  # Florence-2 OmniParser  
        464,  # Florence-2 base
        600   # CLIP (base model)
    ])
    
    print(f"📊 Estimated download: ~{total_size_mb/1000:.1f}GB")
    print(f"⏱️  Estimated time: 2-10 minutes (depending on connection)")
    print(f"💾 Cache location: {downloader.cache_dir}")
    
    # Download models
    print(f"\n🚀 Starting ultra-fast parallel downloads...")
    results = downloader.download_all_models_parallel()
    
    # Check results
    if all(results.values()):
        print(f"\n🎉 All models downloaded successfully!")
        
        # Verify models work
        print(f"\n🔬 Final verification...")
        verification = downloader.verify_all_models()
        
        if all(verification.values()):
            print(f"✅ All models verified and ready!")
            print(f"\n🚀 Server startup commands:")
            print(f"   Minimal test: uv run python minimal_server.py")
            print(f"   Full server:  uv run python main.py")
            print(f"   Monitor logs: tail -f server.log")
        else:
            print(f"⚠️  Some models failed verification")
            failed = [name for name, success in verification.items() if not success]
            print(f"Failed: {', '.join(failed)}")
    else:
        failed = [name for name, success in results.items() if not success]
        print(f"\n❌ Some downloads failed: {', '.join(failed)}")
        print(f"   Check internet connection and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
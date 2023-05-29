import sys
import subprocess
import platform

def is_mac():
    return platform.system() == 'Darwin'

def install():
    
    try:
        import torch
        print(f"PyTorch {torch.__version__} is already installed, skipping installation.")
        return
    except ImportError:
        print("PyTorch is not installed, installing now...")
        if is_mac():
            packages = ["torch", "torchvision", "torchaudio"]
        else:
            #subprocess.check_call([sys.executable, "-m", "pip", "install", "light-the-torch"])
            try:
                subprocess.run(["ltt install torch torchvision torchaudio"], check=True, shell=True)
            except ImportError:
                print("light-the-torch could not be imported. Please install it manually.")
                return

        for package in packages:
            subprocess.check_call(["pip", "install", package])

if __name__ == '__main__':
    install()

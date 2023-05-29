import sys
import subprocess
import platform

def is_mac():
    return platform.system() == 'Darwin'

def install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    if is_mac():
        packages = ["torch", "torchvision", "torchaudio"]
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "light-the-torch"])
        try:
            import light_the_torch as ltt
            packages = ltt.find_links(["torch", "torchvision", "torchaudio"])
        except ImportError:
            print("light-the-torch could not be imported. Please install it manually.")
            return

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':
    install()

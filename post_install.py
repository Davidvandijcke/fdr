# post_install.py
import sys
import subprocess
import platform

def is_mac():
    return platform.system() == 'Darwin'

def install():
    if is_mac():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip3", "install", "torch", "torchvision", "torchaudio"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "light-the-torch"])
        import light_the_torch as ltt
        req = ltt.find_links(["torch", "torchvision", "torchaudio"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + req)

if __name__ == '__main__':
    install()
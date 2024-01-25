from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.run(["python3", "post_install.py"], check=True)
        
install_requires = [
            'numpy',
            #"light-the-torch==0.3.5",
            'scikit-learn',
            'scipy',
            'pywavelets',
            "ray[tune]>=2.8.0",
            "pandas"
        ]

setup(
    name="FDR",
    version="0.1",
    package_dir={'': 'src'},  # Tell Python to look for packages in src/
    packages=find_packages(where='src'),  # Find packages in src/
    python_requires=">=3.6,<3.11", # ray doesn't support 3.11 as of now
    install_requires=install_requires,
        package_data={
        'FDR': ['models/*.pt'],
    },
        cmdclass={
        'install': PostInstallCommand,
    },
)
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

class BuildFromSource(build_py):
    def finalize_options(self):
        super().finalize_options()
        # The historical repository contains build/lib. Archive checkouts
        # can give those copies the same timestamp as the package sources.
        self.force = True
        
install_requires = [
            'numpy',
            'torch',
            'matplotlib',
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
        'build_py': BuildFromSource,
    },
)

from setuptools import setup, find_packages

setup(
    name="sarah",
    version="1.0.0",
    description="Scale-Adaptive Offline Handwritten Text Recognition Framework",
    author="A. Venkata Balaji",
    author_email="galigastagiramma@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fonttools",
        "levenshtein",
        "matplotlib",
        "mlflow",
        "opencv-python-headless",
        "pandas",
        "tensorflow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion"
    ]
)

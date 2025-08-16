from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="nanocrystal-builder",
    version="0.1.0",
    description="Wulff-construction nanocrystal builder with coordination-aware surface passivation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Ivan Infante",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy>=1.11",
        "pyyaml",
        "pymatgen>=2023.7",
    ],
    entry_points={
        "console_scripts": [
            "nc-builder=builder.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    project_urls={
        "Source": "https://github.com/your-user/nanocrystal-builder",
        "Issues": "https://github.com/your-user/nanocrystal-builder/issues",
    },
)



from pathlib import Path

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Optional beam-key C extension.
# Prefer Cythonizing the .pyx when Cython is installed; otherwise compile a
# pre-generated ``_beam_key.c`` if present.  Import-time still falls back to
# pure Python (``_beam_key_fallback``) when no shared library is available.
_BEAM_KEY_PYX = ROOT / "src/builder/nucleation/_beam_key.pyx"
_BEAM_KEY_C = ROOT / "src/builder/nucleation/_beam_key.c"
ext_modules: list = []
if _BEAM_KEY_PYX.is_file():
    try:
        from Cython.Build import cythonize

        ext_modules = cythonize(
            [
                Extension(
                    "builder.nucleation._beam_key",
                    [str(_BEAM_KEY_PYX.relative_to(ROOT))],
                    extra_compile_args=["-O3"],
                )
            ],
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
            annotate=False,
        )
    except Exception:
        ext_modules = []
if not ext_modules and _BEAM_KEY_C.is_file():
    # Cython not installed: compile the checked-in / previously generated C.
    ext_modules = [
        Extension(
            "builder.nucleation._beam_key",
            [str(_BEAM_KEY_C.relative_to(ROOT))],
            extra_compile_args=["-O3"],
        )
    ]

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
        "networkx>=3.0",
        "scipy>=1.11",
        "pyyaml",
        "pymatgen>=2023.7",
    ],
    extras_require={
        "speed": ["cython>=3.0"],
        "test": ["pytest>=7.4"],
    },
    ext_modules=ext_modules,
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
)

"""Testing wheel (nki-library-testing): test framework + pytest plugin.

Packages test/utils/ as nkilib_testing via package_dir remapping.
Depends on nki-library for kernel code. Provides a pytest plugin
(auto-registered via pytest11) with fixtures for zero-config testing.
"""

from build_utils import get_version
from setuptools import find_packages, setup

version = get_version()

# Discover packages under test/utils/ and remap to nkilib_testing namespace
test_util_packages = find_packages(where="test", include=["utils", "utils.*"])
testing_packages = [p.replace("utils", "nkilib_testing", 1) for p in test_util_packages]

setup(
    name="nki_library_testing",
    version=version,
    description="NKI Library Testing Framework",
    install_requires=[
        f"nki-library=={version}",
        "pytest>=8.0",
        "numpy>=1.21",
        "torch>=1.13",
        "boto3>=1.26",
        "paramiko>=2.11",
        "fabric2>=2.0",
        "filelock>=3.0",
        "allpairspy>=2.0",
        "typing_extensions>=4.0",
        "aws-embedded-metrics>=3.0",
        "plotext>=5.0",
        "pandas>=1.3",
    ],
    packages=testing_packages,
    package_dir={"nkilib_testing": "test/utils"},
    package_data={"nkilib_testing": ["bin/*"]},
    entry_points={
        "pytest11": ["nkilib_testing = nkilib_testing.pytest_plugin"],
    },
)

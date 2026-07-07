"""Testing wheel (nki-library-testing): test framework + pytest plugin + test files.

Packages:
  - test/utils/        → nkilib_testing (test framework)
  - test/integration/  → nkilib_testing.tests (test files)

Import rewrites (test.utils.* → nkilib_testing.*) happen at build time
via custom build_py command. Source files stay untouched.
"""

# Prevent pyproject.toml [project].name from overriding our setup() name
import setuptools.config.pyprojecttoml as _pyprojecttoml  # noqa: E402, I001

_pyprojecttoml.apply_configuration = lambda dist, *a, **kw: dist

from build_utils import get_version
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class RewriteImportsBuildPy(build_py):
    """Rewrite test.utils.* → nkilib_testing.* during wheel build."""

    def build_module(self, module, module_file, package):
        outfile, copied = super().build_module(module, module_file, package)

        # Rewrite imports in test files
        if outfile.endswith('.py') and 'nkilib_testing' in outfile:
            with open(outfile, 'r') as f:
                content = f.read()
            original = content
            content = content.replace('from test.utils.', 'from nkilib_testing.')
            content = content.replace('from test.utils import', 'from nkilib_testing import')
            content = content.replace('import test.utils.', 'import nkilib_testing.')
            content = content.replace('from test.integration.nkilib.', 'from nkilib_testing.tests.')
            content = content.replace('import test.integration.nkilib.', 'import nkilib_testing.tests.')
            if content != original:
                with open(outfile, 'w') as f:
                    f.write(content)

        return outfile, copied


version = get_version()

# Discover packages under test/utils/ and remap to nkilib_testing namespace
test_util_packages = find_packages(where="test", include=["utils", "utils.*"])
testing_packages = [p.replace("utils", "nkilib_testing", 1) for p in test_util_packages]

# Discover test file packages under test/integration/nkilib/ and remap to nkilib_testing.tests
test_file_packages = find_packages(where="test/integration", include=["nkilib", "nkilib.*"])
test_packages = [f"nkilib_testing.tests{p[len('nkilib') :]}" for p in test_file_packages]

all_packages = testing_packages + test_packages

setup(
    name="nki_library_testing",
    version=version,
    description="NKI Library Testing Framework + Test Files",
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
    packages=all_packages,
    package_dir={
        "nkilib_testing": "test/utils",
        "nkilib_testing.tests": "test/integration/nkilib",
    },
    package_data={"nkilib_testing": ["bin/*"]},
    include_package_data=True,
    cmdclass={'build_py': RewriteImportsBuildPy},
    entry_points={
        "pytest11": ["nkilib_testing = nkilib_testing.pytest_plugin"],
    },
)

from setuptools import setup, find_packages


# Get requirements from requirements.txt, stripping the version tags
def open_requirements(path):
    with open(path) as f:
        requires = [r.split("/")[-1] if r.startswith("git+") else r for r in f.read().splitlines()]
    return requires


requires = open_requirements("requirements.txt")

setup(
    name='Helix',
    version='0.0.0',
    url='https://gitlab.etp.kit.edu/delight/helix',
    author='Helix Developers',
    author_email='email@gmail.com',
    description='Strax-based data-processing framework for DELight and MMC R&D.',
    packages=find_packages(),
    install_requires=requires
)

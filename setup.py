import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

setuptools.setup(
    name='saltax',
    version='0.0.0',
    install_requires=requires,
    packages=setuptools.find_packages(),
)

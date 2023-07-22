import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

# Get the long description from the README file
with open('README.md') as file:
    readme = file.read()

setuptools.setup(
    name='saltax',
    author='Lanqing Yuan',
    description='Salting analysis framework for XENONnT.',
    long_description=readme,
    version='0.0.0',
    install_requires=requires,
    setup_requires=['pytest-runner'],
    tests_require=requires+['pytest',"hypothesis","boltons"],
    python_requires=">=3.9",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

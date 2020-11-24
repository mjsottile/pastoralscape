from setuptools import setup, find_packages

setup(
    name='livestock-agent-model',
    version='0-0-0',
    description="Agent-based modeling for households with livestock",
    package_dir={'': '.'},
    packages=find_packages('.'),
    install_requires=[
        'matplotlib>=2',
        'pandas',
        'xlrd>=1.1'
    ]
)


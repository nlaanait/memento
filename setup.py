from setuptools import setup, find_packages

setup(
    name='memento',
    version='0.01',
    packages=['memento'],
    url='',
    license='',
    author='Numan Laanait, Max Pasini Lupo',
    author_email='laanaitn@ornl.gov, lupopasinim@ornl.gov',
    description='',
    install_requires=['torch', 'spinup', 'mpi4py'],
    python_requires='>=3.6',
    package_dir = {'memento': 'memento'},
    # package_data = {'qcdenoise': ['data']},
    # include_package_data=True
)

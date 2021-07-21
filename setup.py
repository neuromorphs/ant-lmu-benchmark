from setuptools import setup

setup(
    name='lmu_benchmark',
    packages=['lmu_benchmark'],
    version='0.0.1',
    description='Evaluation of an LMU on many tasks',
    url='https://github.com/neuromorphs/ant-lmu-benchmark',
    license='GPLv3',
    install_requires=['pytry', 'nengo'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ]
    )

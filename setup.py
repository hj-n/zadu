from setuptools import setup, find_packages
 
setup(
    name                = 'metrics4mdp',
    version             = '0.1',
    description         = 'Quantitative metrics of Multidimensional Projections (MDP) in Python',
    author              = 'Hyeon Jeon',
    author_email        = 'hj@hcil.snu.ac.kr',
    url                 = 'https://github.com/hj-n/metrics4mdp',
    download_url        = 'https://github.com/hj-n/metrics4mdp',
    install_requires    =  [numpy, scipym sklearn, numba],
    packages            = find_packages(exclude = []),
    keywords            = ['MDP', 'Multidimensional Projections', "Dimensionality Reduction", "metric"],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)


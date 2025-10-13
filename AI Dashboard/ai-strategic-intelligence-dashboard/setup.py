from setuptools import setup, find_packages

setup(
    name='ai-strategic-intelligence-dashboard',
    version='0.1.0',
    author='ParameshBaba',
    author_email='parameshbaba5@gmail.com',
    description='A dashboard for AI strategic intelligence insights.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'statsmodels',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
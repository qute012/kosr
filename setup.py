from setuptools import setup, find_packages

setup(
    name='kosr',
    version='latest',
    description='Korean speech recognition based on trasformer transducer on online.',
    author='DeokJin Seo',
    author_email='406023@naver.com',
    url='https://github.com/qute012/Korean-Online-Speech-Recognition',
    packages = find_packages(exclude = ['.ipynb', 'Ksponspeech']),
    install_requires=[
        'torch>=1.7.0',
        'python-Levenshtein',
        'torchaudio == 0.7.0',
    ],
    python_requires='>=3.6'
)

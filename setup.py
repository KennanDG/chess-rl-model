from setuptools import setup, find_packages

setup(
    name='ChessRL',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'gymnasium',
        'numpy',
        'python-chess',
    ],
    description='A custom Gymnasium environment for training RL agents on Chess.',
    author='Kennan Gauthier',
    author_email='kennan.d.gauthier@gmail.com',
    url='https://github.com/KennanDG/chess-rl-model',  # Replace with your repo link
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
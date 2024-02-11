from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """
    Get project dependencies from requirements.txt file.
    """

    requirements = []

    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("/n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Portfolio_Optimization',
    packages=find_packages(),
    version='0.1.0',
    description='This is a project about maximizing the returns of stock market by optimizing the portfolio and minimizing the risks by applying risk management techniques.',
    author='Sujal Luhar',
    author_email='luharsujal2712@gmail.com',
    license='MIT',
    install_requires=get_requirements('requirements.txt'),
)

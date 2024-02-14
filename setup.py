from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Phishing',
    version='0.0.1',
    author='Akram Mubeen',
    author_email='amubeen457@gmail.com',
    install_requires=get_requirements('requirements.txt'),  # Pass the list of requirements
    packages=find_packages()
)

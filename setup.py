from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt', 'r') as f:
        req = f.readlines()
    req = [r.rstrip('\n') for r in req]
    return req

print(find_packages())
setup(
    name="simplement",
    version="0.1.0",
    # description="",
    # author="",
    # python_requires=">=3.10",
    # install_requires=get_requirements(),
    packages=find_packages(),
    # packages=find_packages(include=['src', 'src.*']),
    # packages=find_packages(where="src"),
    # package_dir={"": "src"},
    # entry_points={
    #     "console_scripts": [
    #         "train-model = my_project.cli:train",
    #         "eval-model  = my_project.cli:evaluate",
    #     ],
    # },
)

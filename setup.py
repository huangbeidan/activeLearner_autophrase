import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="active_learner", # Replace with your own username
    version="1.1.0",
    author="Beidan Huang",
    author_email="huangbeidan@gmail.com",
    description="Active Learner for Phrase Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huangbeidan/activeLearner_autophrase",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
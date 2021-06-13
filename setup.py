from setuptools import setup, find_packages

setup(
    name='textrl',
    version='0.0.4',
    description='TextRL - use reinforcement learning to adjust text generation results.',
    url='https://github.com/voidful/TextRL',
    author='Voidful',
    author_email='voidful.stack@gmail.com',
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    setup_requires=['setuptools-git'],
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    license="Apache",
    keywords='transformer huggingface nlp generation reinforcement learning deep learning',
    packages=find_packages(),
    install_requires=[
        "pfrl",
        "gym",
        "transformers"
    ],
    python_requires=">=3.5.0",
    zip_safe=False,
)

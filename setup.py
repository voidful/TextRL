from setuptools import setup, find_packages

setup(
    name='textrl',
    version='1.0.0',
    description='TextRL - reinforcement learning for text generation, built on HuggingFace TRL.',
    url='https://github.com/voidful/TextRL',
    author='Voidful',
    author_email='voidful.stack@gmail.com',
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    setup_requires=['setuptools-git'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    license="Apache",
    keywords='transformer huggingface nlp generation reinforcement learning rlhf ppo grpo dpo kto',
    packages=find_packages(exclude=['tests', 'tests.*', 'example', 'example.*']),
    install_requires=[
        "trl>=0.12.0",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "datasets>=2.21.0",
        "torch>=2.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "quant":   ["bitsandbytes>=0.43.0"],
        "vllm":    ["vllm>=0.6.0"],
        "rewards": ["evaluate>=0.4.0", "rouge-score", "sacrebleu"],
        "dev":     ["pytest>=8", "pytest-xdist", "ruff", "mypy"],
    },
    entry_points={
        'console_scripts': [
            'textrl-train=textrl.cli:train_main',
            'textrl-merge=textrl.cli:merge_main',
            'textrl-eval=textrl.cli:eval_main',
            'textrl-dump=textrl.cli:_deprecated_dump',
        ]
    },
    python_requires=">=3.10",
    zip_safe=False,
)

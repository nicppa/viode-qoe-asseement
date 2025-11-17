"""
视频质量评估系统 - 安装脚本
Video QoE Assessment System - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# 核心依赖（可通过pip安装）
# 系统级依赖（mininet, pyshark/tshark）需要单独安装
core_requirements = [
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'xgboost>=1.5.0',
    'lightgbm==3.3.0',
    'joblib>=1.1.0',
    'PyYAML>=5.4',
    'rich>=10.0.0',
    'python-dateutil>=2.8.0',
]

# 可选依赖
extras_require = {
    'deep-learning': ['torch>=1.10.0'],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'pylint>=2.12.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
}

setup(
    name="video-qoe-assessment",
    version="1.0.0",
    author="BMad",
    author_email="research@video-qoe.org",
    description="基于Mininet的视频质量评估研究工具 | Mininet-based Video QoE Assessment Research Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bmad-research/video-qoe-assessment",
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Networking :: Monitoring",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'vqoe-monitor=scripts.monitor:main',
            'vqoe-train=scripts.train_model:main',
            'vqoe-extract-features=scripts.extract_features:main',
            'vqoe-evaluate=scripts.evaluate_model:main',
        ],
    },
    include_package_data=True,
    package_data={
        'video_qoe': ['configs/*.yaml'],
    },
    zip_safe=False,
)


from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="radiate-ai",
    version="0.1.0",
    author="Ganesh Reddy",
    author_email="b.ganesh.reddy.05@gmail.com",
    description="The fastest way to add RAG to any Python app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ganesh-reddy005/radiate-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "qdrant-client>=1.7.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.0",
        "PyPDF2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    keywords="rag retrieval llm openai embeddings vector-database semantic-search ai ml",
    project_urls={
        "Bug Reports": "https://github.com/Ganesh-reddy005/radiate-ai/issues",
        "Source": "https://github.com/Ganesh-reddy005/radiate-ai",
    },
)

from setuptools import setup, find_packages

setup(
    name="backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "pymysql",
        "python-dotenv",
        "langchain",
        "langchain-experimental",
        "openai",
        "chromadb",
        "tiktoken",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "httpx",
        ],
    },
) 

from setuptools import setup, find_packages

setup(
    name="backend",
    version="0.1.0",
    packages=find_packages(exclude=['tests', 'tests.*', 'scripts', 'cache', '.pytest_cache']),
    install_requires=[
        "langchain>=0.3.0,<0.4.0",
        "langchain-experimental>=0.3.0,<0.4.0",
        "langchain-community>=0.3.0,<0.4.0",
        "langchain-core>=0.3.0,<0.4.0",
        "langchain-openai>=0.0.5",
        "openai>=1.0.0",
        "chromadb>=0.4.22",
        "sqlalchemy>=2.0.0",
        "pymysql>=1.1.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "slowapi>=0.1.8",
        "python-jose[cryptography]>=3.3.1",
        "passlib>=1.7.4",
        "python-multipart>=0.0.6",
        "bcrypt>=4.1.2",
        "itsdangerous>=2.1.2",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
        ],
    },
    python_requires=">=3.11",
) 

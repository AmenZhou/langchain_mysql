brew install python
python3 -m venv langchain_env
source langchain_env/bin/activate

pip install langchain langchain-openai sqlalchemy pymysql langchain-experimental openai

deactivate  # Exit virtual environment (if active)
rm -rf langchain_env  # Delete the virtual environment
python3 -m venv langchain_env  # Recreate venv
source langchain_env/bin/activate  # Activate new venv
pip install --upgrade pip setuptools wheel  # Upgrade tools
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

pip install "pydantic>=1.9,<2.5"
pip install "pydantic>=1.9,<2.5" --only-binary :all:

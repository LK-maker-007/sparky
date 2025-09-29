#!/bin/bash
# create_venv_and_run.sh
# Usage: bash create_venv_and_run.sh

set -e

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment created and dependencies installed."
echo "To activate: source venv/bin/activate"
echo "To run the bot: python main.py"

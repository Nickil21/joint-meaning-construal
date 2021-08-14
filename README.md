# Detecting Joint Meaning Construal by Language and Gesture

This repository contains the code for my GSoC 2021 project in joint collaboration with Red Hen Labs and FrameNet Brazil.

## Data:

We use the [PATS](https://chahuja.com/pats/) dataset and use a subset of the speakers in our experiments.

## Run:

Install library dependencies (preferably using Python 3.8.10):

    # Creating a virtual environment
    python3 -m venv .env
    # Activating a virtual environment
    source .env/bin/activate
    # Install necessary libraries
    pip install -r requirements.txt
    # Optional
    export PYTHONPATH=${PWD}

Finally, execute:

    python app.py
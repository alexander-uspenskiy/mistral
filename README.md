# Mistral Chat Interface

This project provides an interactive chat interface for the mistralai/Mistral-Small-24B-Instruct-2501 model using PyTorch and the Hugging Face Transformers library.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- An Apple Silicon device (optional, for MPS support)

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/alexander-uspenskiy/mistral.git
    cd mistral
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install torch transformers
    ```

4. Set your Hugging Face Hub token:
    ```sh
    export HUGGINGFACE_HUB_TOKEN=your_token_here
    ```

## Usage

Run the chat interface:
```sh
python mistral.py
```

## Features

- Interactive chat interface with the mistralai/Mistral-Small-24B-Instruct-2501 model.
- Progress indicator while generating responses.
- Supports Apple Silicon GPU (MPS) for faster inference.


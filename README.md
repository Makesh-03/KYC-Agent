---
title: KYC Agent
emoji: 📚
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
short_description: customer verification
---

## Setup

This application requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) to be installed on your system to process image-based documents.

### Installation

1.  **Install Tesseract:**
    If you are using macOS, you can install Tesseract using [Homebrew](https://brew.sh/):
    ```bash
    brew install tesseract
    ```
    For other operating systems, please refer to the [official Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).

2.  **Install Python Dependencies:**
    Once Tesseract is installed, you can install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the application, run the following command:
```bash
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

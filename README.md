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

This application is designed to run on Hugging Face Spaces and requires the Tesseract OCR engine for image processing.

### Dependencies

-   **System Packages:** The required system-level dependencies, including Tesseract, are listed in the `packages.txt` file. The Hugg-   **Python Packages:** The necessary Python libraries are listed in the `requirements.txt` file and will also be installed automatically.

### Canada Post API Key

The application uses the Canada Post AddressComplete API to verify the extracted address. To use this feature, you will need to:

1.  **Get an API Key:** Obtain a free development key from the [Canada Post Developer Program](https://www.canadapost-postescanada.ca/cpc/en/business/ecommerce/development/addresscomplete.page).
2.  **Add it to your Space:** In your Hugging Face Space, go to the **"Settings"** tab and add a new **"Secret"**. Name the secret `CANADA_POST_API_KEY` and paste your API key as the value.

## Running the Application

The application will start automatically when the Hugging Face Space is launched.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

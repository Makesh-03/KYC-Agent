import os
import mimetypes
import re
import requests
import gradio as gr
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer, util

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image

# Load sentence transformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")
CANADA_POST_API_KEY = os.getenv("CANADA_POST_API_KEY", "MG59-MX89-EE34-ZR95")

# Handle both PDFs and images
def extract_text_from_file(file_path):
    try:
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type == "application/pdf":
            elements = partition_pdf(file_path)
        elif mime_type and mime_type.startswith("image/"):
            elements = partition_image(filename=file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or image.")

        return "\n".join([str(e) for e in elements])
    except Exception as e:
        if "OCRAgent" in str(e):
            raise RuntimeError(
                "OCR processing failed. Please ensure Tesseract is installed and "
                "that the 'unstructured[local-inference]' package is installed."
            )
        raise e

# Extract Canadian address from text
def extract_address(text):
    pattern = r'\d{1,5} [A-Za-z0-9 .]+, [A-Za-z\'\- ]+, [A-Z]{2}, [A-Z]\d[A-Z] ?\d[A-Z]\d'
    match = re.search(pattern, text)
    return match.group(0) if match else ""

# Semantic similarity between expected and extracted address
def semantic_match(extracted, expected, threshold=0.85):
    embeddings = model.encode([extracted, expected], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

# Validate using Canada Post AddressComplete API
def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return None
    try:
        url = "https://ws1.addresscomplete.com/Rest/Find/v2.10/json3.ws"
        response = requests.get(url, params={
            "Key": CANADA_POST_API_KEY,
            "Text": address,
            "Country": "CAN"
        })
        data = response.json()
        return len(data.get("Items", [])) > 0
    except Exception as e:
        print(f"Canada Post API error: {e}")
        return False

# Main KYC verification function
def kyc_verify(file, expected_address):
    if file is None:
        return {"error": "Please upload a document to verify."}
    try:
        file_path = file.name  # Use the path from the Gradio File object
        text = extract_text_from_file(file_path)
        extracted_address = extract_address(text)

        if not extracted_address:
            return {"error": "No valid Canadian address found in the document."}

        sim_score, sem_ok = semantic_match(extracted_address, expected_address)
        cp_ok = verify_with_canada_post(extracted_address) if CANADA_POST_API_KEY else None

        result = {
            "extracted_address": extracted_address,
            "semantic_similarity": round(sim_score, 3),
            "address_match": sem_ok,
            "canada_post_verified": cp_ok,
            "final_result": sem_ok and (cp_ok if cp_ok is not None else True)
        }
        return result

    except Exception as e:
        return {"error": str(e)}

# Gradio UI
iface = gr.Interface(
    fn=kyc_verify,
    inputs=[
        gr.File(label="Upload Passport or Bill (PDF or Image)", file_types=["pdf", "image"]),
        gr.Textbox(label="Expected Address", placeholder="e.g., 123 Main St, Toronto, ON, A1A1A1")
    ],
    outputs="json",
    title="🇨🇦 KYC Document Verifier",
    description="Upload a Canadian document and verify the extracted address against the expected input using OCR and NLP."
)

if __name__ == "__main__":
    iface.launch()

import os
import gradio as gr
import re
import requests
from tempfile import NamedTemporaryFile
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")
CANADA_POST_API_KEY = os.getenv("CANADA_POST_API_KEY", "")

def extract_text_from_pdf(file_path):
    elements = partition_pdf(file_path)
    return "\n".join([str(e) for e in elements])

def extract_address(text):
    pattern = r'\d{1,5} [A-Za-z ]+, [A-Za-z]+, [A-Z]{2}, [A-Z]\d[A-Z] \d[A-Z]\d'
    match = re.search(pattern, text)
    return match.group(0) if match else ""

def semantic_match(extracted, expected, threshold=0.85):
    embeddings = model.encode([extracted, expected], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return False
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
        print(f"Canada Post error: {e}")
        return False

def kyc_verify(file, expected_address):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        file_path = tmp.name

    text = extract_text_from_pdf(file_path)
    extracted_address = extract_address(text)

    if not extracted_address:
        return {"error": "No address found in document."}

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

# Gradio Interface
iface = gr.Interface(
    fn=kyc_verify,
    inputs=[
        gr.File(label="Upload Passport/Bill (PDF)"),
        gr.Textbox(label="Expected Address")
    ],
    outputs="json",
    title="🇨🇦 KYC Document Verifier",
    description="Upload a Canadian document and verify the extracted address against the expected input using OCR and NLP."
)

if __name__ == "__main__":
    iface.launch()
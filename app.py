import os
import mimetypes
import time
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceEndpoint

# --- Model and API Configuration ---

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Mistral via Hugging Face Inference Endpoint
HF_TOKEN = os.getenv("HF_TOKEN")
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=100,
    temperature=0.2,
)

CANADA_POST_API_KEY = os.getenv("CANADA_POST_API_KEY")

# --- Core Functions ---

def extract_text_from_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == "application/pdf":
        elements = partition_pdf(file_path)
    elif mime_type and mime_type.startswith("image/"):
        elements = partition_image(filename=file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image.")
    return "\n".join([str(e) for e in elements])

def extract_address_with_llm(text):
    prompt = PromptTemplate(
        template=(
            "Extract the full Canadian mailing address from the following text. "
            "Include street, city, province, and postal code.\n\n"
            "Text: {document_text}\n\nAddress:"
        ),
        input_variables=["document_text"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"document_text": text})
    return result["text"].strip()

def semantic_match(extracted, expected, threshold=0.85):
    embeddings = similarity_model.encode([extracted, expected], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return {"error": "CANADA_POST_API_KEY secret not set in Hugging Face Space settings."}
    url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
    response = requests.get(
        url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
    )
    data = response.json()
    return len(data.get("Items", [])) > 0

def kyc_verify(file, expected_address):
    if file is None:
        return {"error": "Please upload a document to verify."}
    try:
        results = {}
        t0 = time.time()

        # Step 1: Text Extraction
        text = extract_text_from_file(file.name)
        if not text:
            return {"error": "Could not extract any text from the document."}
        t1 = time.time()
        results["text_extraction_time"] = round(t1 - t0, 2)

        # Step 2: LLM Address Extraction
        extracted_address = extract_address_with_llm(text)
        if not extracted_address:
            return {"error": "Could not find a valid address in the document."}
        t2 = time.time()
        results["llm_extraction_time"] = round(t2 - t1, 2)

        # Step 3: Semantic Match
        sim_score, sem_ok = semantic_match(extracted_address, expected_address)
        t3 = time.time()
        results["semantic_match_time"] = round(t3 - t2, 2)

        # Step 4: Canada Post API
        cp_ok = verify_with_canada_post(extracted_address)
        t4 = time.time()
        results["canada_post_time"] = round(t4 - t3, 2)

        # Final Results
        results.update({
            "extracted_address": extracted_address,
            "semantic_similarity": round(sim_score, 3),
            "address_match": sem_ok,
            "canada_post_verified": cp_ok,
            "final_result": sem_ok and (cp_ok if cp_ok is not None else True),
            "total_time": round(t4 - t0, 2),
        })
        return results
    except Exception as e:
        return {"error": str(e)}

# --- Custom CSS for Styling ---

custom_css = """
h1 {
    font-size: 42px !important;
    font-weight: 900 !important;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.purple-circle {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    background-color: #a020f0 !important;
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    font-size: 18px;
    font-weight: bold;
    margin-right: 10px;
}
.gr-textbox label, .gr-file label {
    font-size: 18px !important;
    font-weight: bold;
}
.purple-button button,
.purple-button button:hover,
.purple-button button:focus {
    background-color: #a020f0 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: none !important;
    transition: background 0.3s ease-in-out;
}
"""

# --- Gradio Interface ---

with gr.Blocks(css=custom_css, title="EZOFIS KYC Agent") as iface:
    gr.Markdown("# EZOFIS KYC Agent")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>1</span> **Upload Document**")
            file_input = gr.File(
                label="Upload Passport or Bill (PDF or Image)",
                file_types=["pdf", "image"]
            )
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>2</span> **Enter Expected Address**")
            expected_address = gr.Textbox(
                label="Expected Address",
                placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>3</span> **KYC Verification Results**")
            output_json = gr.JSON(label="Verification Output")

    with gr.Row():
        verify_btn = gr.Button("🔍 Verify Now", elem_classes="purple-button")

    verify_btn.click(
        fn=kyc_verify,
        inputs=[file_input, expected_address],
        outputs=output_json
    )

if __name__ == "__main__":
    iface.launch()

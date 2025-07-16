# Full updated KYC Agent with Canada Post Authenticity Score

import os
import re
import json
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# --- Config ---
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CANADA_POST_API_KEY = os.getenv("CANADA_POST_API_KEY")

# --- Helpers ---
def filter_non_null_fields(data):
    return {k: v for k, v in data.items() if v not in [None, "null", "", "None"]}

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        elements = partition_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        elements = partition_image(filename=file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image.")
    return "\n".join([str(e) for e in elements])

def get_llm(model_choice="OpenAI"):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    return ChatOpenAI(
        temperature=0.2,
        model_name="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=2000,
    )

def clean_extracted_address(raw_response, original_text=""):
    flattened = raw_response.replace("\n", ", ").replace("  ", " ").strip()
    flattened = re.sub(r"^\s*(\d+[\.\-\):]?)\s*", "", flattened)

    match = re.search(
        r"\d{1,5}[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        flattened,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()

    fallback = re.search(
        r"\d{1,5}[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        original_text.replace("\n", " "),
        re.IGNORECASE,
    )
    if fallback:
        return fallback.group(0).strip()

    return flattened

def extract_address_with_llm(text, model_choice="OpenAI"):
    template = (
        "Extract the full Canadian residential address from the scanned document text. "
        "Include only the house number, street name, city, province (2-letter code), and postal code. "
        "Do not make up or guess any part. Use digits as-is. Avoid decimals. "
        "Example: 145 BAY STREET TORONTO, ON M5J 2R7\n\n"
        "Text:\n{document_text}\n\nExtracted Address:"
    )
    prompt = PromptTemplate(template=template, input_variables=["document_text"])
    llm = get_llm(model_choice)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"document_text": text})
    raw = result["text"].strip()
    cleaned = clean_extracted_address(raw, original_text=text)
    return cleaned

def extract_kyc_fields(text, model_choice="OpenAI"):
    prompt_text = """
You are an expert KYC document parser. Extract all relevant information from the provided document, regardless of whether it is a passport, license, visa, etc. Return ONLY the resulting JSON object. If any field is missing, set it as null.

{
  "document_type": "string or null",
  "document_number": "string or null",
  "country_of_issue": "string or null",
  "issuing_authority": "string or null",
  "full_name": "string or null",
  "first_name": "string or null",
  "middle_name": "string or null",
  "last_name": "string or null",
  "gender": "string or null",
  "date_of_birth": "string or null",
  "place_of_birth": "string or null",
  "nationality": "string or null",
  "address": "string or null",
  "date_of_issue": "string or null",
  "date_of_expiry": "string or null",
  "blood_group": "string or null",
  "personal_id_number": "string or null",
  "father_name": "string or null",
  "mother_name": "string or null",
  "marital_status": "string or null",
  "photo_base64": "string or null",
  "signature_base64": "string or null",
  "additional_info": "string or null"
}

Text:
{text}
"""
    llm = get_llm(model_choice)
    prompt = PromptTemplate(template=prompt_text, input_variables=["text"], template_format="f-string")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"text": text})
    raw_output = result["text"].strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]+\}', raw_output)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                return {"error": "Failed to parse cleaned JSON block"}
        else:
            return {"error": "No JSON block found in response"}

def semantic_match(text1, text2, threshold=0.82):
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return False, None, 0

    url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
    response = requests.get(
        url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
    )
    data = response.json()
    items = data.get("Items", [])

    if not items:
        return False, None, 0

    top_result = items[0].get("Text", "")
    score, _ = semantic_match(address, top_result)
    return True, top_result, int(round(score * 100))

def kyc_multi_verify(files, expected_address, model_choice="OpenAI"):
    if not files or len(files) < 2:
        return "❌ Verification Failed: Please upload at least two documents.", {}, {}

    try:
        extracted_addresses = []
        kyc_data = {}
        similarity_scores = []
        canada_post_verifications = []
        suggested_addresses = []
        authenticity_scores = []

        for idx, file in enumerate(files[:3]):
            text = extract_text_from_file(file.name)
            address = extract_address_with_llm(text, model_choice)
            sim_score, match = semantic_match(address, expected_address)
            verified, suggested_address, authenticity_score = verify_with_canada_post(address)
            kyc_fields = filter_non_null_fields(extract_kyc_fields(text, model_choice))

            extracted_addresses.append(address)
            similarity_scores.append((sim_score, match))
            canada_post_verifications.append(verified)
            suggested_addresses.append(suggested_address)
            authenticity_scores.append(authenticity_score)
            kyc_data[f"document_{idx+1}"] = kyc_fields

        consistency_score, consistent = semantic_match(extracted_addresses[0], extracted_addresses[1])
        percent_score = int(round(consistency_score * 100))
        passed = all(m for _, m in similarity_scores) and all(canada_post_verifications) and consistent

        verification_result = {
            "extracted_addresses": extracted_addresses,
            "suggested_canada_post_addresses": suggested_addresses,
            "authenticity_scores": authenticity_scores,
            "similarity_scores_to_expected": [round(s, 3) for s, _ in similarity_scores],
            "address_matches": [m for _, m in similarity_scores],
            "canada_post_verified": canada_post_verifications,
            "document_consistency_score": round(consistency_score, 3),
            "documents_consistent": consistent,
            "final_result": passed
        }

        status = f"✅ <b style='color:green;'>Verification Passed</b><br>Consistency Score: <b>{percent_score}%</b>" if passed else f"❌ <b style='color:red;'>Verification Failed</b><br>Consistency Score: <b>{percent_score}%</b>"

        return status, verification_result, kyc_data

    except Exception as e:
        return f"❌ <b style='color:red;'>Error:</b> {str(e)}", {}, {}

# UI definition remains unchanged — connect this logic to your existing Gradio layout

# --- UI ---
custom_css = """
.purple-small {
    background-color: #a020f0 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 13px !important;
    padding: 4px 14px !important;
    border-radius: 5px !important;
    border: none !important;
}

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
"""

with gr.Blocks(css=custom_css, title="EZOFIS KYC Agent") as iface:
    gr.Markdown("# EZOFIS KYC Agent")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>1</span> **Upload 2 or 3 Documents (e.g., License, Cheque)**")
            multi_file_input = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg"], label="KYC Documents", file_count="multiple")
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>2</span> **Enter Expected Address**")
            expected_address = gr.Textbox(
                label="Expected Address",
                placeholder="e.g., 145 BAY STREET TORONTO, ON M5J 2R7"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>3</span> **LLM Provider (OpenAI GPT-4o only)**")
            model_choice = gr.Textbox(value="OpenAI", visible=False)

            verify_btn = gr.Button("🔍 Verify Now", elem_classes="purple-small")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>4</span> **KYC Verification Status**")
            status_html = gr.HTML()

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>5</span> **KYC Verification Details**")
            details = gr.Accordion("View Full Verification Details", open=False)
            with details:
                output_json = gr.JSON(label="KYC Output")
                with gr.Group():
                    gr.Markdown("### Extracted Document Details")
                    document_info_json = gr.JSON(label="Document Fields")

    verify_btn.click(
        fn=kyc_multi_verify,
        inputs=[multi_file_input, expected_address, model_choice],
        outputs=[status_html, output_json, document_info_json]
    )

if __name__ == "__main__":
    iface.launch(share=True)

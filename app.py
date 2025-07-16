import gradio as gr
import os
import json
import re
import fitz  # PyMuPDF
import requests
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
CANADA_POST_API_KEY = os.getenv("CANADA_POST_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === UTILS ===
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    else:
        elements = partition(filename=file_path)
        return "\n".join([el.text for el in elements if el.text.strip() != ""])

# === MISTRAL PROMPT FOR ADDRESS EXTRACTION ===
address_prompt_template = (
    "You are an expert information extractor. Your task is to extract ONLY the full Canadian mailing address "
    "from official government-issued identity documents such as a driver’s license, passport, or utility bill. "
    "The address must include:\n"
    "- House or building number (e.g., 2, 742, 8.2)\n"
    "- Street name\n"
    "- City\n"
    "- Province (e.g., ON, NL)\n"
    "- Postal code (format: A1A 1A1)\n\n"
    "⚠️ Do NOT skip the house/building number.\n"
    "⚠️ Ignore section labels like '8.', '9.', 'Eyes:', 'Class:', etc.\n"
    "✅ Return only the full address in one line. No explanation, no labels.\n"
    "✅ Example output: 742 Evergreen Terrace, Ottawa, ON K1A 0B1\n\n"
    "Text:\n{document_text}\n\nExtracted Address:"
)

def call_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"].strip()

def extract_address_with_llm(text, provider):
    prompt = address_prompt_template.format(document_text=text)
    if provider == "Mistral":
        return call_mistral(prompt)
    elif provider == "OpenAI":
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"].strip()

# === POST-CLEANUP ===
def clean_address(addr):
    match = re.search(r"\d{1,5}(?:[.-]?\d+)?[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2}\s+[A-Z]\d[A-Z][ ]?\d[A-Z]\d", addr, re.IGNORECASE)
    return match.group(0).strip() if match else addr.strip()

# === SEMANTIC MATCH ===
def semantic_match(a, b):
    emb1, emb2 = model.encode([a, b], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2).item())

# === CANADA POST ===
def verify_with_canada_post(address):
    return True  # Placeholder logic; assume success

# === MAIN LOGIC ===
def kyc_multi_verify(files, expected_address, provider):
    extracted_addresses = []
    doc_fields = {}

    try:
        for i, file in enumerate(files):
            text = extract_text_from_file(file.name)
            raw_addr = extract_address_with_llm(text, provider)
            clean_addr = clean_address(raw_addr)
            extracted_addresses.append(clean_addr)
            doc_fields[f"document_{i+1}"] = {"address": clean_addr}

        sims = [semantic_match(addr, expected_address) for addr in extracted_addresses]
        matches = [expected_address.lower() in addr.lower() for addr in extracted_addresses]
        post_checks = [verify_with_canada_post(addr) for addr in extracted_addresses]

        # Document consistency (pairwise match avg)
        score = sum(
            semantic_match(a1, a2)
            for i, a1 in enumerate(extracted_addresses)
            for j, a2 in enumerate(extracted_addresses)
            if i < j
        )
        total_pairs = len(extracted_addresses) * (len(extracted_addresses) - 1) / 2
        consistency_score = round(score / total_pairs, 3) if total_pairs else 1.0

        final_result = all(matches) and all(post_checks) and consistency_score > 0.85

        results = {
            **{f"extracted_address_{i+1}": a for i, a in enumerate(extracted_addresses)},
            **{f"similarity_to_expected_{i+1}": round(s, 3) for i, s in enumerate(sims)},
            **{f"address_match_{i+1}": m for i, m in enumerate(matches)},
            **{f"canada_post_verified_{i+1}": p for i, p in enumerate(post_checks)},
            "document_consistency_score": consistency_score,
            "documents_consistent": consistency_score > 0.85,
            "final_result": final_result
        }

        status_msg = (
            f"<div style='color:green;font-size:20px'><b>✅ Address Match Successful ({round(consistency_score*100)}%)</b></div>"
            if final_result else
            f"<div style='color:red;font-size:20px'><b>❌ Address Verification Failed</b></div>"
        )

        return status_msg, results, doc_fields

    except Exception as e:
        return f"❌ Error: {str(e)}", {}, {}

def extract_kyc_fields(text, model_choice):
    template = """
You are an expert KYC document parser. Extract only factual data from the document.
If any field is missing, set it to "Not provided". DO NOT infer.

The address must include building/house number, street, city, province, postal code.

Return only the JSON below:

{{
  "document_type": "string or 'Not provided'",
  "document_number": "string or 'Not provided'",
  "country_of_issue": "string or 'Not provided'",
  "issuing_authority": "string or 'Not provided'",
  "full_name": "string or 'Not provided'",
  "first_name": "string or 'Not provided'",
  "middle_name": "string or 'Not provided'",
  "last_name": "string or 'Not provided'",
  "gender": "string or 'Not provided'",
  "date_of_birth": "string or 'Not provided'",
  "place_of_birth": "string or 'Not provided'",
  "nationality": "string or 'Not provided'",
  "address": "string or 'Not provided'",
  "date_of_issue": "string or 'Not provided'",
  "date_of_expiry": "string or 'Not provided'",
  "blood_group": "string or 'Not provided'",
  "personal_id_number": "string or 'Not provided'",
  "father_name": "string or 'Not provided'",
  "mother_name": "string or 'Not provided'",
  "marital_status": "string or 'Not provided'",
  "photo_base64": "string or 'Not provided'",
  "signature_base64": "string or 'Not provided'",
  "additional_info": "string or 'Not provided'"
}}

Text:
{text}
"""
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    prompt = PromptTemplate(template=template, input_variables=["text"], template_format="f-string")
    result = LLMChain(llm=get_llm(model_choice), prompt=prompt).invoke({"text": text})
    raw_output = result["text"].strip()
    try:
        return json.loads(raw_output)
    except Exception:
        json_match = re.search(r"\{[\s\S]+\}", raw_output)
        try:
            return json.loads(json_match.group()) if json_match else {
                "document_type": "Not provided",
                "document_number": "Not provided",
                "country_of_issue": "Not provided",
                "issuing_authority": "Not provided",
                "full_name": "Not provided",
                "first_name": "Not provided",
                "middle_name": "Not provided",
                "last_name": "Not provided",
                "gender": "Not provided",
                "date_of_birth": "Not provided",
                "place_of_birth": "Not provided",
                "nationality": "Not provided",
                "address": "Not provided",
                "date_of_issue": "Not provided",
                "date_of_expiry": "Not provided",
                "blood_group": "Not provided",
                "personal_id_number": "Not provided",
                "father_name": "Not provided",
                "mother_name": "Not provided",
                "marital_status": "Not provided",
                "photo_base64": "Not provided",
                "signature_base64": "Not provided",
                "additional_info": "Not provided",
                "error": "No JSON block found",
                "raw_output": raw_output
            }
        except Exception:
            return {
                "document_type": "Not provided",
                "document_number": "Not provided",
                "country_of_issue": "Not provided",
                "issuing_authority": "Not provided",
                "full_name": "Not provided",
                "first_name": "Not provided",
                "middle_name": "Not provided",
                "last_name": "Not provided",
                "gender": "Not provided",
                "date_of_birth": "Not provided",
                "place_of_birth": "Not provided",
                "nationality": "Not provided",
                "address": "Not provided",
                "date_of_issue": "Not provided",
                "date_of_expiry": "Not provided",
                "blood_group": "Not provided",
                "personal_id_number": "Not provided",
                "father_name": "Not provided",
                "mother_name": "Not provided",
                "marital_status": "Not provided",
                "photo_base64": "Not provided",
                "signature_base64": "Not provided",
                "additional_info": "Not provided",
                "error": "Failed to parse KYC fields",
                "raw_output": raw_output
            }

# --- UI ---
custom_css = """
.purple-small {
    background-color: #a020f0 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 8px 20px !important;
    border-radius: 6px !important;
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
.purple-button button {
    background-color: #a020f0 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(css=custom_css, title="EZOFIS KYC Agent") as iface:
    gr.Markdown("# EZOFIS KYC Agent")
    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>1</span> **Upload Documents (2 or more)**")
            file_inputs = gr.File(
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                file_count="multiple",
                label="Documents"
            )
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>2</span> **Enter Expected Address**")
            expected_address = gr.Textbox(
                label="Expected Address",
                placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2"
            )
            gr.Markdown("<span class='purple-circle'>3</span> **Select LLM Provider**")
            model_choice = gr.Dropdown(
                choices=["Mistral", "OpenAI"],
                value="Mistral",
                label="LLM Provider"
            )
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
                gr.Markdown("### Extracted Document Details")
                document_info_json = gr.JSON(label="Document Fields")

    verify_btn.click(
        fn=kyc_multi_verify,
        inputs=[file_inputs, expected_address, model_choice],
        outputs=[status_html, output_json, document_info_json]
    )

if __name__ == "__main__":
    iface.launch(share=True)

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

def filter_non_null_fields(data):
    return {k: v for k, v in data.items() if v not in [None, "null", "", "None", "Not provided"]}

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        elements = partition_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        elements = partition_image(filename=file_path)
    else:
        raise ValueError("Unsupported file type.")
    return "\n".join([str(e) for e in elements])

def get_llm(model_choice):
    model_map = {
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "OpenAI": "openai/gpt-4o"
    }
    return ChatOpenAI(
        temperature=0.2,
        model_name=model_map[model_choice],
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=2000,
    )

def clean_address_mistral(raw_response, original_text=""):
    # Flatten and clean up initial formatting
    flattened = raw_response.replace("\n", ", ").replace("  ", " ").strip()
    
    # Remove section prefixes like '8.', '8)', '8:', '8.2', '8a)', etc.
    flattened = re.sub(r"^(?:\s*(\d{1,2}(?:\.\d+)?[\.\):a-zA-Z]?\s*)+)", "", flattened)
    
    # Remove common misleading prefixes like "Section 8", "8.", "8)", "8.2"
    flattened = re.sub(r"(?i)(?:section\s*\d{1,2}(?:\.\d+)?[\.\):]?\s*|item\s*\d{1,2}[\.\):]?\s*)", "", flattened)

    # Enhanced regex for Canadian address:
    # - Building number: 1-5 digits, optionally followed by a single letter (e.g., "123", "38A")
    # - Street name: At least one word, optionally followed by a street type (e.g., "Main St", "Evergreen")
    # - City: One or more words
    # - Province: 2-letter code
    # - Postal code: A1A 1A1
    match = re.search(
        r"(\d{1,5}[A-Za-z]?\s+[\w\s'-]+(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln))?,?\s*[\w\s'-]+,?\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d)",
        flattened,
        re.IGNORECASE,
    )
    if match:
        extracted_address = match.group(0).strip()
        # Validate semantically against a template address
        template_address = "123 Main Street, Toronto, ON M5V 2N2"
        sim_score, is_valid = semantic_match(extracted_address, template_address, threshold=0.7)
        if is_valid:
            return extracted_address

    # Fallback: stricter regex on original text
    fallback = re.search(
        r"(\d{1,5}[A-Za-z]?\s+[\w\s'-]+(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln))?,?\s*[\w\s'-]+,?\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d)",
        original_text.replace("\n", " "),
        re.IGNORECASE,
    )
    if fallback:
        extracted_address = fallback.group(0).strip()
        # Validate fallback address semantically
        template_address = "123 Main Street, Toronto, ON M5V 2N2"
        sim_score, is_valid = semantic_match(extracted_address, template_address, threshold=0.7)
        if is_valid:
            return extracted_address

    # If no match, return cleaned flattened text
    return flattened

def extract_address_with_llm(text, model_choice):
    if model_choice == "Mistral":
        template = (
            "You are a strict document parser extracting Canadian addresses.\n\n"
            "Your task is to extract ONLY the full Canadian mailing address from the document text. The address must include:\n"
            "- A house or building number (e.g., '123', '38A', NOT a section number like '8.', '8.2', or '8)')\n"
            "- Street name (e.g., 'Main Street', 'Evergreen Terrace')\n"
            "- City or town (e.g., 'Toronto', 'St. John's')\n"
            "- Province (use two-letter code like ON, NL)\n"
            "- Postal code (format: A1A 1A1)\n\n"
            "**IMPORTANT RULES:**\n"
            "- DO NOT include section numbers (e.g., '8.', '8.2', '9.', '8)', '8a)') or labels like 'Section 8', 'Item 8', 'Eyes:', 'Class:'.\n"
            "- The address must begin with the actual building number (e.g., '2 Thorburn Road', NOT '8.2 Thorburn Road').\n"
            "- Ignore any numbers that appear to be part of document sections or labels.\n"
            "- If multiple addresses exist, pick the one that is clearly a Canadian枢 for Canadian residential or mailing address.\n"
            "- Return ONLY the address in one line. No extra words, explanations, or labels.\n\n"
            "Example Input:\n"
            "8.2 Thorburn Road, St. John's, NL A1B 3M2\n"
            "Section 8: 742 Evergreen Terrace, Ottawa, ON K1A 0B1\n"
            "9) 123 Main St, Toronto, ON M5V 2N2\n\n"
            "Example Output:\n"
            "742 Evergreen Terrace, Ottawa, ON K1A 0B1\n\n"
            "Text:\n{document_text}\n\nExtracted Address:"
        )
    else:
        template = (
            "Extract the full Canadian mailing address from the following text. "
            "Include street, city, province, and postal code.\n\n"
            "Text: {document_text}\n\nAddress:"
        )
    prompt = PromptTemplate(template=template, input_variables=["document_text"])
    chain = LLMChain(llm=get_llm(model_choice), prompt=prompt)
    result = chain.invoke({"document_text": text})
    return clean_address_mistral(result["text"].strip(), original_text=text) if model_choice == "Mistral" else result["text"].strip()

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
                "error": "Failed to parses KYC fields",
                "raw_output": raw_output
            }

def semantic_match(text1, text2, threshold=0.82):
    embeddings = similarity_model.encode([text1, text)2], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return False
    url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
    response = requests.get(
        url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
    )
    return len(response.json().get("Items", [])) > 0

def kyc_multi_verify(files, expected_address, model_choice):
    if not files or len(files) < 2:
        return "❌ Please upload at least two documents.", {}, {}
    try:
        results = {}
        kyc_fields = {}
        addresses = []
        for idx, file in enumerate(files):
            text = extract_text_from_file(file.name)
            address = extract_address_with_llm(text, model_choice)
            sim, match = semantic_match(address, expected_address)
            verified = verify_with_canada_post(address)
            fields = extract_kyc_fields(text, model_choice)
            addresses.append(address)
            results[f"extracted_address_{idx+1}"] = address
            results[f"similarity_to_expected_{idx+1}"] = round(sim, 3)
            results[f"address_match_{idx+1}"] = match
            results[f"canada_post_verified_{idx+1}"] = verified
            kyc_fields[f"document_{idx+1}"] = filter_non_null_fields(fields)

        consistency_score, consistent = semantic_match(addresses[0], addresses[1])
        results["document_consistency_score"] = round(consistency_score, 3)
        results["documents_consistent"] = consistent
        results["final_result"] = all([results[f"address_match_{i+1}"] and results[f"canada_post_verified_{i+1}"] for i in range(len(files))]) and consistent

        status = (
            f"✅ <b style='color:green;'>Verification Passed</b><br>Consistency Score: <b>{int(round(consistency_score * 100))}%</b>"
            if results["final_result"]
            else f"❌ <b style='color:red;'>Verification Failed</b><br>Consistency Score: <b>{int(round(consistency_score * 100))}%</b>"
        )
        return status, results, kyc_fields
    except Exception as e:
        return f"❌ Error: {str(e)}", {}, {}

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
    color:  white !important;
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
            file_inputs = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg"], file_count="multiple", label="Documents")
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>2</span> **Enter Expected Address**")
            expected_address = gr.Textbox(label="Expected Address", placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2")
            gr.Markdown("<span class='purple-circle'>3</span> **Select LLM Provider**")
            model_choice = gr.Dropdown(choices=["Mistral", "OpenAI"], value="Mistral", label="LLM Provider")
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
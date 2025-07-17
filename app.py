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
    flattened = raw_response.replace("\n", ", ").replace("  ", " ").strip()
    # Minimal filtering to remove section headers, preserve building number
    flattened = re.sub(r"^(?:\s*(\d{1,2}(?:\.\d+)?[\.\):])\s*)+", "", flattened)
    flattened = re.sub(r"(?i)section\s*\d{1,2}(?:\.\d+)?[\.\):]?\s*", "", flattened)
    # Check for basic address structure, fall back to original text if needed
    if re.search(r"\d+\s+[\w\s,]+,\s*\w+,\s*[A-Z]{2},\s*[A-Z]\d[A-Z]\s*\d[A-Z]\d", flattened, re.IGNORECASE):
        return flattened
    fallback = re.search(
        r"(?:Apt|Unit|Suite|Door)?\s*\d{1,5}\s+[\w\s,.-]+,\s*\w+,\s*[A-Z]{2},\s*[A-Z]\d[A-Z]\s*\d[A-Z]\d",
        original_text.replace("\n", " "),
        re.IGNORECASE,
    )
    return fallback.group(0).strip() if fallback else flattened

def extract_address_with_llm(text, model_choice):
    if model_choice == "Mistral":
        template = (
            "You are an expert document parser. Extract the full Canadian mailing address from the document text, including the building/house number (e.g., '2', 'Apt 2', 'Door 3'), street name, city, province (two-letter code like ON, NL), and postal code (format: A1A 1A1). The building number must be included, even if it appears after the street name or in a non-standard format; ignore section headers (e.g., '8.', '8.2', '9)'). Return only the address in one line, no extra text or notes.\n\n"
            "Example Output:\nDoor 2, 123 Main St, St. John’s, NL A1A 1A1\n2 Thorburn Road, St. John’s, NL A1B 3L7\n\n"
            "Text:\n{document_text}\n\nExtracted Address:"
        )
    else:
        template = (
            "Extract the full Canadian mailing address from the following text. "
            "Include street, city, province, and postal code. Ensure the building number is a standalone number (e.g., '2', 'Apt 2', 'Door 3', not '8.2').\n\n"
            "Text: {document_text}\n\nAddress:"
        )
    prompt = PromptTemplate(template=template, input_variables=["document_text"])
    chain = LLMChain(llm=get_llm(model_choice), prompt=prompt)
    result = chain.invoke({"document_text": text})
    print(f"Raw LLM response for {model_choice}: {result['text'].strip()}")  # Debug print
    return clean_address_mistral(result["text"].strip(), original_text=text)

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
        fields = json.loads(raw_output)
        if "address" in fields and fields["address"] != "Not provided":
            fields["address"] = clean_address_mistral(fields["address"], original_text=text)
        return fields
    except Exception:
        json_match = re.search(r"\{[\s\S]+\}", raw_output)
        try:
            fields = json.loads(json_match.group()) if json_match else {
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
            if "address" in fields and fields["address"] != "Not provided":
                fields["address"] = clean_address_mistral(fields["address"], original_text=text)
            return fields
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

def semantic_match(text1, text2, threshold=0.82):
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return False, "No API key provided"
    url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
    response = requests.get(
        url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
    )
    items = response.json().get("Items", [])
    if items:
        verified_address = items[0].get("Text", address)
        return True, verified_address
    return False, address

def kyc_multi_verify(files, expected_address, model_choice, consistency_threshold):
    if not files or len(files) < 2:
        return "❌ Please upload at least two documents.", {}, {}
    try:
        results = {}
        kyc_fields = {}
        addresses = []
        authenticity_scores = []
        for idx, file in enumerate(files):
            text = extract_text_from_file(file.name)
            address = extract_address_with_llm(text, model_choice)
            sim, match = semantic_match(address, expected_address)
            verified, canada_post_address = verify_with_canada_post(address)
            auth_score, _ = semantic_match(address, canada_post_address)
            authenticity_scores.append(auth_score)
            fields = extract_kyc_fields(text, model_choice)
            addresses.append(address)
            results[f"extracted_address_{idx+1}"] = address
            results[f"similarity_to_expected_{idx+1}"] = round(sim, 3)
            results[f"address_match_{idx+1}"] = match
            results[f"canada_post_verified_{idx+1}"] = verified
            results[f"authenticity_score_{idx+1}"] = round(auth_score, 3)
            kyc_fields[f"document_{idx+1}"] = filter_non_null_fields(fields)

        consistency_score, consistent = semantic_match(addresses[0], addresses[1], threshold=consistency_threshold)
        avg_authenticity_score = sum(authenticity_scores) / len(authenticity_scores)
        results["document_consistency_score"] = round(consistency_score, 3)
        results["documents_consistent"] = consistent
        results["average_authenticity_score"] = round(avg_authenticity_score, 3)
        results["final_result"] = all([results[f"address_match_{i+1}"] and results[f"canada_post_verified_{i+1}"] for i in range(len(files))]) and (consistency_score >= consistency_threshold)

        status = (
            f"✅ <b style='color:green;'>Verification Passed</b><br>Consistency Score: <b>{int(round(consistency_score * 100))}%</b><br>Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
            if results["final_result"]
            else f"❌ <b style='color:red;'>Verification Failed</b><br>Consistency Score: <b>{int(round(consistency_score * 100))}%</b><br>Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
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
.gr-textbox label, .gr-file label, .gr-slider label {
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
.gr-slider input[type="range"] {
    -webkit-appearance: none;
    width: 100%;
    height: 10px;
    background: #d3d3d3;
    outline: none;
    border-radius: 5px;
    margin-top: 10px;
}
.gr-slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    background: #a020f0;
    border-radius: 50%;
    cursor: pointer;
}
.gr-slider input[type="range"]::-moz-range-thumb {
    width: 20px;
    height: 20px;
    background: #a020f0;
    border-radius: 50%;
    cursor: pointer;
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
            gr.Markdown("<span class='purple-circle'>4</span> **Set Consistency Threshold**")
            consistency_threshold = gr.Slider(minimum=0.5, maximum=1.0, value=0.82, step=0.01, label="Consistency Threshold")
            verify_btn = gr.Button("🔍 Verify Now", elem_classes="purple-small")
    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>5</span> **KYC Verification Status**")
            status_html = gr.HTML()
    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>6</span> **KYC Verification Details**")
            details = gr.Accordion("View Full Verification Details", open=False)
            with details:
                output_json = gr.JSON(label="KYC Output")
                gr.Markdown("### Extracted Document Details")
                document_info_json = gr.JSON(label="Document Fields")
    verify_btn.click(
        fn=kyc_multi_verify,
        inputs=[file_inputs, expected_address, model_choice, consistency_threshold],
        outputs=[status_html, output_json, document_info_json]
    )

if __name__ == "__main__":
    iface.launch(share=True)
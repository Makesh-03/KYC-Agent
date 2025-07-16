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
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            elements = partition_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            elements = partition_image(filename=file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or image.")
        return "\n".join([str(e) for e in elements])
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return "❌ Error: Unable to extract text. Please check your file and environment."

def get_llm(model_choice):
    model_map = {
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "OpenAI": "openai/gpt-4o"
    }
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    return ChatOpenAI(
        temperature=0.2,
        model_name=model_map[model_choice],
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=2000,
    )

def clean_address_mistral(raw_response, original_text=""):
    flattened = raw_response.replace("\n", ", ").replace("  ", " ").strip()
    flattened = re.sub(r"^\s*(\d+[\.\-\):]?)\s*", "", flattened)

    match = re.search(
        r"\d{1,5}(?:[.\-]?\d+)?[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        flattened,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()

    fallback = re.search(
        r"\d{1,5}(?:[.\-]?\d+)?[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        original_text.replace("\n", " "),
        re.IGNORECASE,
    )
    if fallback:
        return fallback.group(0).strip()
    return flattened

def extract_address_with_llm(text, model_choice):
    if model_choice == "Mistral":
        template = (
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
    else:
        template = (
            "Extract the full Canadian mailing address from the following text. "
            "Include street, city, province, and postal code.\n\n"
            "Text: {document_text}\n\nAddress:"
        )
    prompt = PromptTemplate(template=template, input_variables=["document_text"])
    llm = get_llm(model_choice)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"document_text": text})
    raw = result["text"].strip()
    return clean_address_mistral(raw, original_text=text) if model_choice == "Mistral" else raw

def extract_kyc_fields(text, model_choice, extracted_address_1=None):
    prompt_text = """
You are an expert KYC document parser. Extract all relevant information from the provided document, regardless of whether it is a passport, driving license, national identity card, or any type of VISA document. Return ONLY the resulting JSON object (no explanation). Use this schema:

{{
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
}}

Text:
{text}
"""
    llm = get_llm(model_choice)
    prompt = PromptTemplate(template=prompt_text, input_variables=["text"], template_format="f-string")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"text": text})
    raw_output = result["text"].strip()
    try:
        kyc_output = json.loads(raw_output)
    except Exception:
        json_match = re.search(r'\{[\s\S]+\}', raw_output)
        try:
            kyc_output = json.loads(json_match.group()) if json_match else {"error": "No JSON block found"}
        except Exception:
            kyc_output = {
                "document_type": None,
                "document_number": None,
                "country_of_issue": None,
                "issuing_authority": None,
                "full_name": None,
                "first_name": None,
                "middle_name": None,
                "last_name": None,
                "gender": None,
                "date_of_birth": None,
                "place_of_birth": None,
                "nationality": None,
                "address": None,
                "date_of_issue": None,
                "date_of_expiry": None,
                "blood_group": None,
                "personal_id_number": None,
                "father_name": None,
                "mother_name": None,
                "marital_status": None,
                "photo_base64": None,
                "signature_base64": None,
                "additional_info": None,
                "error": "Failed to parse KYC fields",
                "raw_output": raw_output
            }
    if "address" in kyc_output:
        addr = kyc_output["address"]
        if not addr or len(str(addr)) < 10 or re.search(r"(Eyes|Sex|Height|Classe|\d+\.)", str(addr), re.IGNORECASE):
            kyc_output["address"] = extracted_address_1
    return kyc_output

def semantic_match(text1, text2, threshold=0.82):
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    if not CANADA_POST_API_KEY:
        return False
    url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
    response = requests.get(
        url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
    )
    data = response.json()
    return len(data.get("Items", [])) > 0

def kyc_multi_verify(files, expected_address, model_choice):
    if not files or len(files) < 2:
        return "❌ Upload at least 2 documents.", {}, {}
    try:
        addresses, fields, sims, matches, verifications = [], [], [], [], []
        for file in files:
            text = extract_text_from_file(file.name)
            addr = extract_address_with_llm(text, model_choice)
            kyc = filter_non_null_fields(extract_kyc_fields(text, model_choice))
            sim, matched = semantic_match(addr, expected_address)
            verified = verify_with_canada_post(addr)
            addresses.append(addr)
            sims.append(round(sim, 3))
            matches.append(matched)
            verifications.append(verified)
            fields.append(kyc)

        consistency_score, documents_consistent = semantic_match(addresses[0], addresses[1])
        final_result = all(matches) and all(verifications) and documents_consistent
        verification_result = {
            f"extracted_address_{i+1}": addresses[i] for i in range(len(addresses))
        }
        verification_result.update({
            f"similarity_to_expected_{i+1}": sims[i] for i in range(len(addresses))
        })
        verification_result.update({
            f"address_match_{i+1}": matches[i] for i in range(len(addresses))
        })
        verification_result.update({
            f"canada_post_verified_{i+1}": verifications[i] for i in range(len(addresses))
        })
        verification_result["document_consistency_score"] = round(consistency_score, 3)
        verification_result["documents_consistent"] = documents_consistent
        verification_result["final_result"] = final_result

        kyc_combined = {f"document_{i+1}": fields[i] for i in range(len(fields))}
        status = (
            f"✅ <b style='color:green;'>Verification Passed</b><br>Consistency Score: <b>{int(consistency_score*100)}%</b>"
            if final_result else
            f"❌ <b style='color:red;'>Verification Failed</b><br>Consistency Score: <b>{int(consistency_score*100)}%</b>"
        )
        return status, verification_result, kyc_combined
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
.purple-button button,
.purple-button button:hover,
.purple-button button:focus {
    background-color: #a020f0 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: none !important;
    transition: background 0.3s ease-in-out;
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
                with gr.Group():
                    gr.Markdown("### Extracted Document Details")
                    document_info_json = gr.JSON(label="Document Fields")

    verify_btn.click(
        fn=kyc_multi_verify,
        inputs=[file_inputs, expected_address, model_choice],
        outputs=[status_html, output_json, document_info_json]
    )

if __name__ == "__main__":
    iface.launch(share=True)

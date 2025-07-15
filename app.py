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

# --- Core Functions ---
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        elements = partition_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        elements = partition_image(filename=file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image.")
    return "\n".join([str(e) for e in elements])

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
    flattened = re.sub(r"^\s*(\d+(\.\d+)?[\.\-\):]?)\s*", "", flattened)

    if re.match(r"(?i)(Sexe|Eyes|Class)", flattened):
        flattened = ""

    pattern = r"\b(?<!\d\.)(\d{1,5})[\w\s.,'-]+?,\s*\w+,
                \s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d\b"

    match = re.search(pattern, flattened)
    if match:
        return match.group(0).strip()

    fallback_match = re.search(pattern, original_text.replace("\n", " "))
    if fallback_match:
        return fallback_match.group(0).strip()

    return "Address not found"

def extract_address_with_llm(text, model_choice):
    if model_choice == "Mistral":
        template = (
            "You are extracting address information from Canadian government-issued documents such as a driver\u2019s license. "
            "The address usually includes a house number, street, city, province, and postal code. "
            "Ignore unrelated fields like gender, class, height, eye color, license restrictions. "
            "Do NOT include section numbers like 8, 15, or labels like 'Sexe' or 'Eyes'. "
            "Return only the clean Canadian address. Do not explain anything.\n\n"
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

def extract_kyc_fields(text, model_choice):
    prompt_text = """
You are an expert KYC document parser. Extract all relevant information from the provided document, regardless of whether it is a passport, driving license, national identity card, or any type of VISA document (including but not limited to tourist visas, work visas, student visas, resident visas, entry permits, e-visas, visa labels, or visa stamps). Use the visible field labels, visual structure, and document layout to capture the data, but do not invent fields or infer details that do not explicitly appear on the document. If any field is missing, set its value as null. Return ONLY the resulting JSON object (no explanation, no text before or after). Use this schema for the unified KYC data extraction:

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
    # Use template_format="f-string" to avoid curly brace parsing issues
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
        return False
    url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
    response = requests.get(
        url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
    )
    data = response.json()
    return len(data.get("Items", [])) > 0

def kyc_dual_verify(file1, file2, expected_address, model_choice):
    if file1 is None or file2 is None:
        return "❌ Verification Failed: Please upload both documents.", {}, {}

    try:
        text1 = extract_text_from_file(file1.name)
        text2 = extract_text_from_file(file2.name)
        address1 = extract_address_with_llm(text1, model_choice)
        address2 = extract_address_with_llm(text2, model_choice)

        sim1, match1 = semantic_match(address1, expected_address)
        sim2, match2 = semantic_match(address2, expected_address)
        verified1 = verify_with_canada_post(address1)
        verified2 = verify_with_canada_post(address2)
        consistency_score, consistent = semantic_match(address1, address2)
        percent_score = int(round(consistency_score * 100))

        kyc_fields_1 = filter_non_null_fields(extract_kyc_fields(text1, model_choice))
        kyc_fields_2 = filter_non_null_fields(extract_kyc_fields(text2, model_choice))

        kyc_combined = {
            "first_document": kyc_fields_1,
            "second_document": kyc_fields_2
        }

        passed = all([match1, match2, verified1, verified2, consistent])
        verification_result = {
            "extracted_address_1": address1,
            "extracted_address_2": address2,
            "similarity_to_expected_1": round(sim1, 3),
            "similarity_to_expected_2": round(sim2, 3),
            "address_match_1": match1,
            "address_match_2": match2,
            "canada_post_verified_1": verified1,
            "canada_post_verified_2": verified2,
            "document_consistency_score": round(consistency_score, 3),
            "documents_consistent": consistent,
            "final_result": passed
        }

        if passed:
            status = f"✅ <b style='color:green;'>Verification Passed</b><br>Consistency Score: <b>{percent_score}%</b>"
        else:
            status = f"❌ <b style='color:red;'>Verification Failed</b><br>Consistency Score: <b>{percent_score}%</b>"

        return status, verification_result, kyc_combined

    except Exception as e:
        return f"❌ <b style='color:red;'>Error:</b> {str(e)}", {}, {}


# --- UI ---
with gr.Blocks(title="EZOFIS KYC Agent") as iface:
    gr.Markdown("# EZOFIS KYC Agent")

    with gr.Row():
        with gr.Column():
            file_input_1 = gr.File(label="Upload Document 1", file_types=[".pdf", ".jpg", ".jpeg", ".png"])
        with gr.Column():
            expected_address = gr.Textbox(label="Expected Address")

    with gr.Row():
        with gr.Column():
            file_input_2 = gr.File(label="Upload Document 2", file_types=[".pdf", ".jpg", ".jpeg", ".png"])
        with gr.Column():
            model_choice = gr.Dropdown(choices=["Mistral", "OpenAI"], value="Mistral", label="LLM Provider")
            verify_btn = gr.Button("🔍 Verify Now")

    status_html = gr.HTML()
    output_json = gr.JSON(label="Verification Result")
    document_info_json = gr.JSON(label="Extracted Document Fields")

    verify_btn.click(
        fn=kyc_dual_verify,
        inputs=[file_input_1, file_input_2, expected_address, model_choice],
        outputs=[status_html, output_json, document_info_json]
    )

if __name__ == "__main__":
    iface.launch(share=True)
import os
import re
import json
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import time
import mimetypes

# --- Config ---
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
UNSTRACT_API_KEY = os.getenv("UNSTRACT_API_KEY")
UNSTRACT_BASE = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

def filter_non_null_fields(data):
    return {k: v for k, v in data.items() if v not in [None, "null", "", "None", "Not provided"]}

def extract_text_from_file(file_path):
    # Use Unstract Whisperer API (with API key) for PDF and images
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    headers = {
        "unstract-key": UNSTRACT_API_KEY,
        "Content-Type": mimetypes.guess_type(filename)[0] or "application/octet-stream"
    }
    up = requests.post(
        f"{UNSTRACT_BASE}/whisper",
        headers=headers,
        data=file_bytes,
    )
    if up.status_code not in (200, 202):
        raise RuntimeError(f"OCR upload failed ({up.status_code})")
    token = up.json()["whisper_hash"]

    # Poll /whisper-status up to 3 min
    for poll_sec in range(180):
        time.sleep(1)
        status_resp = requests.get(
            f"{UNSTRACT_BASE}/whisper-status?whisper_hash={token}",
            headers={"unstract-key": UNSTRACT_API_KEY},
        )
        status_json = status_resp.json()
        status = status_json.get("status")
        # Uncomment to debug: print(f"[{poll_sec+1}s] Whisper-status: {status_json}")
        if status == "processed":
            break
        elif status == "failed":
            raise RuntimeError(f"Unstract Whisperer processing failed: {status_json}")
    else:
        raise RuntimeError("Unstract Whisperer API timeout waiting for job completion.")

    # GET /whisper-retrieve
    ret = requests.get(
        f"{UNSTRACT_BASE}/whisper-retrieve?whisper_hash={token}&text_only=true",
        headers={"unstract-key": UNSTRACT_API_KEY},
    )
    try:
        return ret.json().get("result_text") or ret.text
    except Exception as e:
        return ret.text

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
    flattened = re.sub(r"^(?:\s*(\d{1,2}(?:\.\d)?[\.\):])\s*)+", "", flattened)
    flattened = re.sub(r"(?i)section\s*\d{1,2}(?:\.\d)?[\.\):]?\s*", "", flattened)
    flattened = re.sub(r"^\d+\.\d+\s+", "", flattened)
    match = re.search(
        r"^\d{1,5}[A-Za-z\-]?\s+[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        flattened,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()
    fallback = re.search(
        r"^\d{1,5}[A-Za-z\-]?\s+[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        original_text.replace("\n", " "),
        re.IGNORECASE,
    )
    if fallback:
        return fallback.group(0).strip()
    return flattened

def extract_address_with_llm(text, model_choice):
    if model_choice == "Mistral":
        template = (
            "You are a strict document parser extracting Canadian addresses.\n\n"
            "Your task is to extract ONLY the full Canadian mailing address from the document text. The address must include:\n"
            "- A house or building number (must be a standalone number like '2', '742', '38A', NOT a prefixed section number like '8.' or '8)')\n"
            "- Street name\n"
            "- City or town\n"
            "- Province (use two-letter code like ON, NL)\n"
            "- Postal code (format: A1A 1A1)\n\n"
            "**IMPORTANT RULES:**\n"
            "- DO NOT include section numbers (e.g., '8.', '9., '8)', '9)') or labels like 'Eyes:', 'Class:', etc.\n"
            "- Ignore any lines starting with numbers followed by a dot or parenthesis (e.g., '8.', '8.2', '9)') as these are section headers, not addresses."
            "- The address should begin with the actual building number (e.g., '2 Thorburn Road')\n"
            "- Never assume or hallucinate building numbers like '8.2' if the actual number is just '2'.\n"
            "- If multiple addresses exist, pick the one that is clearly a Canadian residential or mailing address.\n\n"
            "Return ONLY the address in one line. No extra words, explanations, or labels.\n\n"
            "Example Output:\n742 Evergreen Terrace, Ottawa, ON K1A 0B1\n\n"
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

# ----- GOOGLE MAPS ADDRESS VALIDATION -----
def verify_with_google_maps(address):
    if not GOOGLE_MAPS_API_KEY:
        return False, "No API key provided"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "region": "ca",  # restricts results to Canada
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get(url, params=params)
    try:
        data = response.json()
        if data.get("status") == "OK" and data.get("results"):
            formatted_address = data["results"][0].get("formatted_address", address)
            return True, formatted_address
        return False, address
    except Exception:
        return False, address

# --- Format verification results as HTML table ---
def format_verification_table(results):
    if not results:
        return ""
    table_html = """
    <div style="border-radius:16px;border:2px solid #A020F0;margin-bottom:32px;background:#D3D3D3;padding:18px 22px 22px 22px;box-shadow:0 3px 16px #0001;">
      <table style="width:100%;border:none;margin-bottom:12px;">
        <tr>
          <td style="width:40%;font-size:17px;font-weight:700;color:#008000;">Final Result:</td>
          <td style="width:60%;font-size:17px;font-weight:700;color:{};">{}</td>
        </tr>
        <tr>
          <td style="font-size:17px;font-weight:700;color:#008000;">Address Consistency Score:</td>
          <td style="font-size:17px;color:#008000;">{}%</td>
        </tr>
        <tr>
          <td style="font-size:17px;font-weight:700;color:#008000;">Name Consistency Score:</td>
          <td style="font-size:17px;color:#008000;">{}%</td>
        </tr>
        <tr>
          <td style="font-size:17px;font-weight:700;color:#008000;">Overall Consistency Score:</td>
          <td style="font-size:17px;color:#008000;">{}%</td>
        </tr>
        <tr>
          <td style="font-size:17px;font-weight:700;color:#008000;">Average Authenticity Score:</td>
          <td style="font-size:17px;color:#008000;">{}%</td>
        </tr>
    """.format(
        "#008000" if results.get("final_result", False) else "#FF0000",
        "Passed" if results.get("final_result", False) else "Failed",
        int(round(results.get("address_consistency_score", 0) * 100)),
        int(round(results.get("name_consistency_score", 0) * 100)),
        int(round(results.get("document_consistency_score", 0) * 100)),
        int(round(results.get("average_authenticity_score", 0) * 100))
    )
    for idx in range(len([k for k in results.keys() if k.startswith("extracted_address_")])):
        table_html += """
        <tr>
          <td style="font-weight:600;font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;color:#008000;">Document {} Address:</td>
          <td style="font-weight:600;font-size:15px;color:#008000;">{}</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;color:#008000;">Document {} Full Name:</td>
          <td style="font-weight:600;font-size:15px;color:#008000;">{}</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;color:#008000;">Address Similarity to Expected:</td>
          <td style="font-weight:600;font-size:15px;color:#008000;">{}%</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;color:#008000;">Address Match:</td>
          <td style="font-weight:600;font-size:15px;color:{};">{}</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;color:#008000;">Google Maps Verified:</td>
          <td style="font-weight:600;font-size:15px;color:{};">{}</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;color:#008000;">Authenticity Score:</td>
          <td style="font-weight:600;font-size:15px;color:#008000;">{}%</td>
        </tr>
        """.format(
            idx + 1,
            results.get(f"extracted_address_{idx+1}", "Not provided"),
            idx + 1,
            results.get(f"extracted_name_{idx+1}", "Not provided"),
            int(round(results.get(f"similarity_to_expected_{idx+1}", 0) * 100)),
            "#008000" if results.get(f"address_match_{idx+1}", False) else "#FF0000",
            "Yes" if results.get(f"address_match_{idx+1}", False) else "No",
            "#008000" if results.get(f"google_maps_verified_{idx+1}", False) else "#FF0000",
            "Yes" if results.get(f"google_maps_verified_{idx+1}", False) else "No",
            int(round(results.get(f"authenticity_score_{idx+1}", 0) * 100))
        )
    table_html += """
      </table>
    </div>
    """
    return table_html

def kyc_multi_verify(files, expected_address, model_choice, consistency_threshold):
    if not files or len(files) < 2:
        return "❌ Please upload at least two documents.", {}, {}
    try:
        results = {}
        kyc_fields = {}
        addresses = []
        names = []
        authenticity_scores = []
        for idx, file in enumerate(files):
            text = extract_text_from_file(file.name)
            address = extract_address_with_llm(text, model_choice)
            fields = extract_kyc_fields(text, model_choice)
            name = fields.get("full_name", "Not provided")
            sim, match = semantic_match(address, expected_address)
            verified, google_maps_address = verify_with_google_maps(address)
            auth_score, _ = semantic_match(address, google_maps_address)
            authenticity_scores.append(auth_score)
            addresses.append(address)
            names.append(name)
            results[f"extracted_address_{idx+1}"] = address
            results[f"extracted_name_{idx+1}"] = name
            results[f"similarity_to_expected_{idx+1}"] = round(sim, 3)
            results[f"address_match_{idx+1}"] = match
            results[f"google_maps_verified_{idx+1}"] = verified
            results[f"authenticity_score_{idx+1}"] = round(auth_score, 3)
            kyc_fields[f"document_{idx+1}"] = filter_non_null_fields(fields)

        address_consistency_score, address_consistent = semantic_match(
            addresses[0], addresses[1], threshold=consistency_threshold
        )
        avg_authenticity_score = sum(authenticity_scores) / len(authenticity_scores)
        results["address_consistency_score"] = round(address_consistency_score, 3)
        # Set name score as info only, does not affect pass/fail
        results["name_consistency_score"] = (
            semantic_match(names[0], names[1])[0]
            if names[0] != "Not provided" and names[1] != "Not provided"
            else 0
        )
        results["document_consistency_score"] = round(address_consistency_score, 3)
        results["documents_consistent"] = address_consistent
        results["average_authenticity_score"] = round(avg_authenticity_score, 3)
        # Final result requires address consistency and all address matches/verifications (names do NOT affect final_result)
        results["final_result"] = all([
            results[f"address_match_{i+1}"] and results[f"google_maps_verified_{i+1}"]
            for i in range(len(files))
        ]) and (address_consistency_score >= consistency_threshold)

        status = (
            f"✅ <b style='color:green;'>Verification Passed</b><br>Address Consistency Score: <b>{int(round(address_consistency_score * 100))}%</b><br>Name Consistency Score: <b>{int(round(results['name_consistency_score'] * 100))}%</b><br>Overall Consistency Score: <b>{int(round(results['document_consistency_score'] * 100))}%</b><br>Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
            if results["final_result"]
            else f"❌ <b style='color:red;'>Verification Failed</b><br>Address Consistency Score: <b>{int(round(address_consistency_score * 100))}%</b><br>Name Consistency Score: <b>{int(round(results['name_consistency_score'] * 100))}%</b><br>Overall Consistency Score: <b>{int(round(results['document_consistency_score'] * 100))}%</b><br>Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
        )
        return status, format_verification_table(results), kyc_fields
    except Exception as e:
        return f"❌ Error: {str(e)}", "", {}

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
                output_html = gr.HTML(label="KYC Output")
                gr.Markdown("### Extracted Document Details")
                document_info_json = gr.JSON(label="Document Fields")
    verify_btn.click(
        fn=kyc_multi_verify,
        inputs=[file_inputs, expected_address, model_choice, consistency_threshold],
        outputs=[status_html, output_html, document_info_json]
    )

if __name__ == "__main__":
    iface.launch(share=True)

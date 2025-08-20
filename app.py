# app.py

import streamlit as st
import requests
import json
import re
import os
import time
import mimetypes
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

# ── MUST be the first Streamlit call ───────────────────────────────────────────
st.set_page_config(page_title="EZOFIS KYC Agent", page_icon="🔍", layout="wide")
# ───────────────────────────────────────────────────────────────────────────────

# --- Config ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# EXACTLY as in your working Gradio reference
UNSTRACT_API_KEY = os.getenv("UNSTRACT_API_KEY")
UNSTRACT_BASE = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

# --- Initialize embedding model only once per session ---
if "similarity_model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        try:
            st.session_state["similarity_model"] = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder="./.cache_sbert",
            )
            st.session_state["similarity_model"].to("cpu")
        except Exception as e:
            st.error(f"Failed to load SentenceTransformer model: {str(e)}")
            st.stop()

def filter_non_null_fields(data):
    return {k: v for k, v in data.items() if v not in [None, "null", "", "None", "Not provided"]}

# ── Unstract OCR: mirrors your Gradio reference (header + flow) ───────────────
def extract_text_from_file(file):
    filename = file.name
    file_bytes = file.read()

    if not UNSTRACT_API_KEY:
        raise RuntimeError("Missing UNSTRACT_API_KEY environment variable.")

    headers = {
        "unstract-key": UNSTRACT_API_KEY,
        "Content-Type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
    }
    # Upload
    up = requests.post(f"{UNSTRACT_BASE}/whisper", headers=headers, data=file_bytes)
    if up.status_code not in (200, 202):
        raise RuntimeError(f"OCR upload failed ({up.status_code}): {up.text}")

    token = up.json().get("whisper_hash")
    if not token:
        raise RuntimeError(f"OCR response missing whisper_hash: {up.text}")

    # Poll /whisper-status up to ~3 minutes
    for _ in range(180):
        time.sleep(1)
        status_resp = requests.get(
            f"{UNSTRACT_BASE}/whisper-status?whisper_hash={token}",
            headers={"unstract-key": UNSTRACT_API_KEY},
        )
        if status_resp.status_code != 200:
            continue
        status_json = status_resp.json()
        status = status_json.get("status")
        if status == "processed":
            break
        elif status == "failed":
            raise RuntimeError(f"Unstract Whisperer processing failed: {status_json}")
    else:
        raise RuntimeError("Unstract Whisperer API timeout waiting for job completion.")

    # Retrieve
    ret = requests.get(
        f"{UNSTRACT_BASE}/whisper-retrieve?whisper_hash={token}&text_only=true",
        headers={"unstract-key": UNSTRACT_API_KEY},
    )
    try:
        return ret.json().get("result_text") or ret.text
    except Exception:
        return ret.text
# ───────────────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(
        temperature=0.2,
        model_name="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=2000,
    )

def clean_address(raw_response, original_text=""):
    flattened = raw_response.replace("\n", ", ").replace(" ", " ").strip()
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

def extract_address_with_llm(text):
    template = (
        "Extract the full Canadian mailing address from the following text. "
        "Include street, city, province, and postal code.\n\n"
        "Text: {document_text}\n\nAddress:"
    )
    prompt = PromptTemplate(template=template, input_variables=["document_text"])
    chain = LLMChain(llm=get_llm(), prompt=prompt)
    result = chain.invoke({"document_text": text})
    return clean_address(result["text"].strip(), original_text=text)

def extract_kyc_fields(text):
    template = """ 
    You are an expert KYC document parser. Extract only factual data from the document. If any field is missing, set it to "Not provided". DO NOT infer. The address must include building/house number, street, city, province, postal code. Return only the JSON below:
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
    Text: {text}
    """
    prompt = PromptTemplate(template=template, input_variables=["text"], template_format="f-string")
    result = LLMChain(llm=get_llm(), prompt=prompt).invoke({"text": text})
    raw_output = result["text"].strip()
    try:
        fields = json.loads(raw_output)
        if "address" in fields and fields["address"] != "Not provided":
            fields["address"] = clean_address(fields["address"], original_text=text)
        return fields
    except Exception:
        json_match = re.search(r"\{[\s\S]+\}", raw_output)
        try:
            fields = (
                json.loads(json_match.group())
                if json_match
                else {
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
                    "raw_output": raw_output,
                }
            )
            if "address" in fields and fields["address"] != "Not provided":
                fields["address"] = clean_address(fields["address"], original_text=text)
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
                "raw_output": raw_output,
            }

def semantic_match(text1, text2, threshold=0.82):
    embeddings = st.session_state["similarity_model"].encode([text1, text2])
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    return sim, sim >= threshold

def verify_with_google_maps(address):
    if not GOOGLE_MAPS_API_KEY:
        return False, "No API key provided"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "region": "ca",
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

# ── Fixed Table builder ──────────────────────────────────────────────────────
def format_verification_table(results):
    if not results:
        return ""

    final_color = "#00ff7f" if results.get("final_result", False) else "#ff4d4d"
    final_text = "Passed" if results.get("final_result", False) else "Failed"

    doc_count = len([k for k in results.keys() if k.startswith("extracted_address_")])

    # Create a clean HTML table without extra indentation
    table_html = f'''<div style="background-color:#111; color:white; border:2px solid #a64dff; padding:16px; border-radius:12px; font-family:Arial, sans-serif; overflow-x:auto;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; border-bottom:1px solid #a64dff; padding-bottom:10px; margin-bottom:16px;">
<span style="font-weight:bold;">Final Result: <span style="color:{final_color};">{final_text}</span></span>
<span><strong>Address Consistency:</strong> {int(results.get("address_consistency_score", 0)*100)}%</span>
<span><strong>Name Consistency:</strong> {int(results.get("name_consistency_score", 0)*100)}%</span>
<span><strong>Overall Consistency:</strong> {int(results.get("document_consistency_score", 0)*100)}%</span>
<span><strong>Avg Authenticity:</strong> {int(results.get("average_authenticity_score", 0)*100)}%</span>
</div>
<table style="width:100%; min-width:600px; border-collapse:collapse; font-size:14px; table-layout:fixed;">
<thead>
<tr style="background-color:#222;">
<th style="padding:10px; border-bottom:2px solid #a64dff; width:5%;">Doc</th>
<th style="padding:10px; border-bottom:2px solid #a64dff; width:25%;">Address</th>
<th style="padding:10px; border-bottom:2px solid #a64dff; width:25%;">Full Name</th>
<th style="padding:10px; border-bottom:2px solid #a64dff; width:10%;">Similarity %</th>
<th style="padding:10px; border-bottom:2px solid #a64dff; width:10%;">Address Match</th>
<th style="padding:10px; border-bottom:2px solid #a64dff; width:10%;">Google Maps</th>
<th style="padding:10px; border-bottom:2px solid #a64dff; width:15%;">Authenticity %</th>
</tr>
</thead>
<tbody>'''

    for idx in range(doc_count):
        address = results.get(f"extracted_address_{idx+1}", "Not provided")
        name = results.get(f"extracted_name_{idx+1}", "Not provided")
        sim_pct = int(round(results.get(f"similarity_to_expected_{idx+1}", 0) * 100))
        match = results.get(f"address_match_{idx+1}", False)
        maps_ok = results.get(f"google_maps_verified_{idx+1}", False)
        auth_pct = int(round(results.get(f"authenticity_score_{idx+1}", 0) * 100))

        match_text = "Yes" if match else "No"
        match_color = "#00ff7f" if match else "#ff4d4d"
        maps_text = "Yes" if maps_ok else "No"
        maps_color = "#00ff7f" if maps_ok else "#ff4d4d"

        table_html += f'''<tr>
<td style="padding:8px; border-bottom:1px solid #333; width:5%; text-align:center;">{idx+1}</td>
<td style="padding:8px; border-bottom:1px solid #333; width:25%; word-break:break-all; overflow-wrap:break-word;">{address}</td>
<td style="padding:8px; border-bottom:1px solid #333; width:25%; word-break:break-word;">{name}</td>
<td style="padding:8px; border-bottom:1px solid #333; width:10%; text-align:center;">{sim_pct}%</td>
<td style="padding:8px; border-bottom:1px solid #333; width:10%; color:{match_color}; font-weight:700; text-align:center;">{match_text}</td>
<td style="padding:8px; border-bottom:1px solid #333; width:10%; color:{maps_color}; font-weight:700; text-align:center;">{maps_text}</td>
<td style="padding:8px; border-bottom:1px solid #333; width:15%; text-align:center;">{auth_pct}%</td>
</tr>'''

    table_html += '''</tbody>
</table>
</div>'''
    
    return table_html

def kyc_multi_verify(files, expected_address, consistency_threshold, name_threshold):
    if not files or len(files) < 2:
        return "❌ Please upload at least two documents.", "", {}

    try:
        results = {}
        addresses = []
        names = []
        authenticity_scores = []
        per_doc_clean_fields = {}

        for idx, file in enumerate(files):
            text = extract_text_from_file(file)
            address = extract_address_with_llm(text)
            fields = extract_kyc_fields(text)
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

            # Keep per-document fields to show in dropdown (cleaned)
            per_doc_clean_fields[f"document{idx+1}"] = filter_non_null_fields(fields)

        # Address consistency between first two docs
        address_consistency_score, address_consistent = semantic_match(
            addresses[0], addresses[1], threshold=consistency_threshold
        )

        # Name consistency using name_threshold
        if names[0] != "Not provided" and names[1] != "Not provided":
            name_consistency_score, name_consistent = semantic_match(
                names[0], names[1], threshold=name_threshold
            )
        else:
            name_consistency_score, name_consistent = (0, False)

        # Average authenticity
        avg_authenticity_score = sum(authenticity_scores) / len(authenticity_scores)

        # Overall score (address + name) average
        overall_score = (address_consistency_score + name_consistency_score) / 2.0

        # Populate results for the table header
        results["address_consistency_score"] = round(address_consistency_score, 3)
        results["name_consistency_score"] = round(name_consistency_score, 3)
        results["document_consistency_score"] = round(address_consistency_score, 3)
        results["documents_consistent"] = address_consistent
        results["average_authenticity_score"] = round(avg_authenticity_score, 3)
        results["overall_score"] = round(overall_score, 3)

        # Final decision logic unchanged
        results["final_result"] = all(
            [results[f"address_match_{i+1}"] and results[f"google_maps_verified_{i+1}"] for i in range(len(files))]
        ) and (address_consistency_score >= consistency_threshold) and name_consistent

        # Status block — Overall Score first
        status = (
            f"✅ <b style='color:green;'>Verification Passed</b><br>"
            f"Overall Score: <b>{int(round(results['overall_score'] * 100))}%</b><br>"
            f"Address Consistency Score: <b>{int(round(address_consistency_score * 100))}%</b><br>"
            f"Name Consistency Score: <b>{int(round(results['name_consistency_score'] * 100))}%</b><br>"
            f"Overall Consistency Score: <b>{int(round(results['document_consistency_score'] * 100))}%</b><br>"
            f"Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
            if results["final_result"]
            else f"❌ <b style='color:red;'>Verification Failed</b><br>"
                 f"Overall Score: <b>{int(round(results['overall_score'] * 100))}%</b><br>"
                 f"Address Consistency Score: <b>{int(round(address_consistency_score * 100))}%</b><br>"
                 f"Name Consistency Score: <b>{int(round(results['name_consistency_score'] * 100))}%</b><br>"
                 f"Overall Consistency Score: <b>{int(round(results['document_consistency_score'] * 100))}%</b><br>"
                 f"Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
        )

        # ── JSON dropdown payload (KYC) ────────────────────────────────────────
        result_block = {
            "Address Consistency Score": f"{int(round(address_consistency_score * 100))}%",
            "Name Consistency Score": f"{int(round(results['name_consistency_score'] * 100))}%",
            "Overall Score": f"{int(round(results['overall_score'] * 100))}%",
            "Overall Consistency Score": f"{int(round(results['document_consistency_score'] * 100))}%",
            "Average Authenticity Score": f"{int(round(avg_authenticity_score * 100))}%"
        }
        data_block = {"result": result_block}
        data_block.update(per_doc_clean_fields)
        json_dropdown_payload = {"data": data_block}
        # Save to session for Debug section
        st.session_state["kyc_debug_json"] = json_dropdown_payload

        # NEW: persist KYC visible outputs
        st.session_state["kyc_status_html"] = status
        st.session_state["kyc_output_html"] = format_verification_table(results)

        return status, st.session_state["kyc_output_html"], json_dropdown_payload

    except Exception as e:
        return f"❌ Error: {str(e)}", "", {}

# --- Streamlit UI ---
def main():
    st.markdown(
        """
<style>
/* Overall dark background */
.stApp { background-color: #121212; color: #ffffff; }

/* Headers */
h1 { font-size: 42px !important; font-weight: 900 !important; color: #ffffff; text-align: center; margin-bottom: 20px; }
h3 { color: #a020f0 !important; font-weight: bold !important; }

/* Purple numbered circle */
.purple-circle { display: inline-flex; justify-content: center; align-items: center; background-color: #a020f0 !important; color: white; border-radius: 50%; width: 40px; height: 40px; font-size: 18px; font-weight: bold; margin-right: 10px; }

/* Input labels */
.stFileUploader > div > div > div > label,
.stTextInput > div > div > label,
.stSelectbox > div > div > label,
.stSlider > div > div > label { font-size: 18px !important; font-weight: bold !important; color: #ffffff !important; }

/* Buttons */
.stButton > button { background-color: #a020f0 !important; color: white !important; font-weight: bold !important; font-size: 16px !important; padding: 10px 22px !important; border-radius: 8px !important; border: none !important; }

/* Sliders */
.stSlider > div > div > div > div { background: #333333 !important; border-radius: 5px; }
.stSlider > div > div > div > div > div { background: #a020f0 !important; border-radius: 50%; }

/* Expanders */
.stExpander { border: 2px solid #a020f0; border-radius: 8px; padding: 10px; background-color: #1e1e1e; color: #ffffff; }

/* Text areas */
.stTextArea > div > div > textarea { background-color: #1e1e1e !important; color: #ffffff !important; border: 1px solid #a020f0 !important; }

/* Markdown outputs inside app */
.stMarkdown, .stText, .stCodeBlock { color: #ffffff !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<h1>EZOFIS KYC Agent</h1>", unsafe_allow_html=True)

    # Main KYC Verification Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<span class='purple-circle'>1</span> <b>Upload Documents (2 or more)</b>", unsafe_allow_html=True)
        file_inputs = st.file_uploader(
            "Documents", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"]
        )

    with col2:
        st.markdown("<span class='purple-circle'>2</span> <b>Enter Expected Address</b>", unsafe_allow_html=True)
        expected_address = st.text_input("Expected Address", placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2")

        st.markdown("<span class='purple-circle'>3</span> <b>Set Address Consistency Threshold</b>", unsafe_allow_html=True)
        consistency_threshold = st.slider("Address Consistency Threshold", min_value=0.5, max_value=1.0, value=0.82, step=0.01)

        st.markdown("<span class='purple-circle'>4</span> <b>Set Name Consistency Threshold</b>", unsafe_allow_html=True)
        name_threshold = st.slider("Name Consistency Threshold", min_value=0.5, max_value=1.0, value=0.80, step=0.01)

    verify_btn = st.button("🔍 Verify Now")

    st.markdown("<span class='purple-circle'>5</span> <b>KYC Verification Status</b>", unsafe_allow_html=True)
    status_placeholder = st.empty()

    st.markdown("<span class='purple-circle'>6</span> <b>KYC Verification Details</b>", unsafe_allow_html=True)
    with st.expander("View Full Verification Details", expanded=False):
        output_placeholder = st.empty()

    # NEW: Rehydrate KYC outputs every run
    if "kyc_status_html" in st.session_state:
        status_placeholder.markdown(st.session_state["kyc_status_html"], unsafe_allow_html=True)
    if "kyc_output_html" in st.session_state:
        output_placeholder.markdown(st.session_state["kyc_output_html"], unsafe_allow_html=True)

    if verify_btn:
        if file_inputs and expected_address and consistency_threshold and name_threshold:
            with st.spinner("Verifying..."):
                status, output_html, _document_info_json = kyc_multi_verify(
                    file_inputs, expected_address, consistency_threshold, name_threshold
                )
                # Also render immediately this run
                status_placeholder.markdown(status, unsafe_allow_html=True)
                output_placeholder.markdown(output_html, unsafe_allow_html=True)
        else:
            st.error(
                "Please provide all required inputs: at least two documents, expected address, and both thresholds."
            )

    # Face Verification Section - MOVED TO THE END
    st.markdown("<hr style='border: 1px solid #a020f0; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3>🧑‍💼 Face Verification</h3>", unsafe_allow_html=True)
    st.markdown("Compare faces from ID document and selfie image using advanced deep learning models.", unsafe_allow_html=True)

    face_col1, face_col2 = st.columns(2)

    with face_col1:
        st.markdown("<span class='purple-circle'>7</span> <b>Upload ID/License Image</b>", unsafe_allow_html=True)
        id_image = st.file_uploader("ID/License Image", type=["png", "jpg", "jpeg"], key="id_upload")

    with face_col2:
        st.markdown("<span class='purple-circle'>8</span> <b>Upload Selfie Image</b>", unsafe_allow_html=True)
        selfie_image = st.file_uploader("Selfie Image", type=["png", "jpg", "jpeg"], key="selfie_upload")

    face_verify_btn = st.button("🔍 Compare Faces")

    st.markdown("<span class='purple-circle'>9</span> <b>Face Verification Results</b>", unsafe_allow_html=True)

    # Always show Face results from session_state (persist across reruns)
    face_message_value = st.session_state.get("face_message", "")
    st.text_area("Face Match Results", face_message_value, height=120, key="face_results_display")

    # Always show cropped faces if available
    if st.session_state.get("face_cropped_id") is not None and st.session_state.get("face_cropped_selfie") is not None:
        st.markdown("### Cropped Face Comparison")
        crop_col1, crop_col2 = st.columns(2)
        with crop_col1:
            st.markdown("**ID/License Face:**")
            st.image(st.session_state["face_cropped_id"], width=200, caption="Extracted from ID document")
        with crop_col2:
            st.markdown("**Selfie Face:**")
            st.image(st.session_state["face_cropped_selfie"], width=200, caption="Extracted from selfie")

    if face_verify_btn:
        if id_image is None or selfie_image is None:
            st.warning("⚠️ Please upload both ID/License image and Selfie image to proceed.")
        else:
            with st.spinner("Analyzing faces..."):
                try:
                    # Import face verification libraries only when needed
                    import cv2
                    import numpy as np
                    from PIL import Image
                    import tempfile

                    # Import DeepFace with error handling
                    try:
                        from deepface import DeepFace
                        import tensorflow as tf
                        # Force TensorFlow to use CPU to avoid GPU-related issues
                        tf.config.set_visible_devices([], 'GPU')
                    except Exception as import_error:
                        st.error(f"❌ Could not import DeepFace library: {str(import_error)}")
                        st.info("💡 Face verification requires the DeepFace library. Please install it with: pip install deepface")
                        st.stop()

                    # ---- ArcFace compatibility probe (require TF 2.15 + Keras 2.x or tf-keras 2.15) ----
                    arcface_ok = False
                    arcface_reason = ""

                    # Check TensorFlow version
                    try:
                        import tensorflow as tf  # noqa: F401
                        tf_ok = str(tf.__version__).startswith("2.15")
                    except Exception as e:
                        tf_ok = False
                        arcface_reason = f"TensorFlow not compatible ({e})"

                    # Check for tf-keras shim and/or Keras 2.x
                    tk_ok = False
                    keras2_ok = False
                    try:
                        import tf_keras as _tk  # noqa: F401
                        tk_ok = True
                    except Exception:
                        pass
                    try:
                        import keras  # noqa: F401
                        keras2_ok = str(keras.__version__).startswith("2.")
                        if not keras2_ok and not arcface_reason:
                            arcface_reason = f"Keras {keras.__version__} detected"
                    except Exception as e:
                        if not arcface_reason:
                            arcface_reason = f"Keras not importable ({e})"

                    arcface_ok = tf_ok and (tk_ok or keras2_ok)

                    if not arcface_ok:
                        st.info(
                            "ℹ️ ArcFace disabled due to incompatible runtime. Requires TensorFlow 2.15.x and either "
                            "Keras 2.x or tf-keras 2.15. Using VGG-Face and Facenet only."
                        )

                    # Load Haar cascade for face detection
                    try:
                        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                        if face_cascade.empty():
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    except Exception as cascade_error:
                        st.error(f"❌ Could not load face detection cascade: {str(cascade_error)}")
                        st.stop()

                    # Face helpers
                    def auto_crop_face(image_pil):
                        if image_pil.mode != "RGB":
                            image_pil = image_pil.convert("RGB")
                        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                        best_crop = None
                        max_area = 0
                        for angle in [0, 90, 180, 270]:
                            rotated = (
                                img_cv if angle == 0 else
                                cv2.rotate(img_cv, {
                                    90: cv2.ROTATE_90_CLOCKWISE,
                                    180: cv2.ROTATE_180,
                                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                                }[angle])
                            )
                            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                            for (x, y, w, h) in faces:
                                area = w * h
                                if area > max_area:
                                    max_area = area
                                    best_crop = rotated[y:y + h, x:x + w]
                        return Image.fromarray(cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)) if best_crop is not None else None

                    def verify_faces(img1_pil, img2_pil, arcface_ok: bool, arcface_reason: str):
                        try:
                            cropped1 = auto_crop_face(img1_pil)
                            cropped2 = auto_crop_face(img2_pil)

                            if cropped1 is None:
                                return "❌ No face detected in License/ID image.", None, None, None
                            if cropped2 is None:
                                return "❌ No face detected in Selfie image.", None, None, None

                            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp1, tempfile.NamedTemporaryFile(suffix=".jpg") as tmp2:
                                cropped1.save(tmp1.name)
                                cropped2.save(tmp2.name)

                                # Build the model trial list dynamically
                                models_to_try = [
                                    ("VGG-Face", "opencv"),
                                    ("Facenet", "opencv"),
                                ]
                                if arcface_ok:
                                    models_to_try.append(("ArcFace", "opencv"))
                                models_to_try.extend([
                                    ("VGG-Face", "mtcnn"),
                                    ("Facenet", "mtcnn"),
                                ])

                                distances = []
                                model_results = []  # structured per-model results

                                debug_info = {}
                                if not arcface_ok and arcface_reason:
                                    debug_info["arcface_note"] = (
                                        f"ArcFace skipped: {arcface_reason}. Requires TF 2.15 + (Keras 2.x or tf-keras 2.15)."
                                    )

                                for model_name, detector in models_to_try:
                                    try:
                                        result = DeepFace.verify(
                                            tmp1.name,
                                            tmp2.name,
                                            model_name=model_name,
                                            detector_backend=detector,
                                            distance_metric="cosine",
                                            enforce_detection=False,
                                            align=True,
                                        )
                                        dist = float(result.get("distance", 1.0))
                                        sim = (1 - dist) * 100
                                        distances.append(dist)
                                        model_results.append({
                                            "model": model_name,
                                            "detector": detector,
                                            "distance": round(dist, 4),
                                            "similarity_percent": round(sim, 2)
                                        })

                                        # Stop early once we have 3 successful signals
                                        if len(distances) >= 3:
                                            break

                                    except Exception:
                                        # Skip failed model entries in results list
                                        continue

                                if not distances:
                                    return "❌ All face recognition models failed. Please try with clearer images.", None, None, None

                                max_similarity = max((1 - d) * 100 for d in distances)
                                avg_similarity = sum((1 - d) * 100 for d in distances) / len(distances)

                                if max_similarity > 60:
                                    verdict = "✅ Match (High Confidence)"
                                elif max_similarity > 45:
                                    verdict = "⚠️ Possible Match (Medium Confidence)"
                                else:
                                    verdict = "❌ No Match"

                                # Display message (ONLY verdict + matching score)
                                display_message = (
                                    f"{verdict}\n"
                                    f"Matching score: {max_similarity:.2f}%"
                                )

                                # Build debug JSON payload (for Debug dropdown)
                                debug_info.update({
                                    "average_similarity_percent": round(avg_similarity, 2),
                                    "successful_models": f"{len(distances)}/{len(models_to_try)}",
                                    "model_results": model_results
                                })

                                return display_message, debug_info, cropped1, cropped2

                        except Exception as e:
                            return (f"❌ Face verification error: {str(e)}\n\n"
                                    f"Tip: Try using clearer images with good lighting."), None, None, None

                    # Process the uploaded images
                    id_img_pil = Image.open(id_image)
                    selfie_img_pil = Image.open(selfie_image)

                    message, face_debug, cropped_id, cropped_selfie = verify_faces(id_img_pil, selfie_img_pil, arcface_ok, arcface_reason)

                    # Persist results to session_state (so they survive reruns)
                    st.session_state["face_message"] = message
                    st.session_state["face_cropped_id"] = cropped_id
                    st.session_state["face_cropped_selfie"] = cropped_selfie

                    # Save face debug JSON to session for Debug dropdown at the end
                    if face_debug is not None:
                        st.session_state["face_debug_json"] = {"data": {"face_verification": face_debug}}

                    # Also render immediately this run
                    st.session_state["face_message"] = message  # already set
                    st.experimental_rerun()  # ensures the always-on widgets above show latest
                except Exception as e:
                    st.error(f"❌ Error processing images: {str(e)}")

    # ── DEBUG SECTION MOVED TO THE VERY END (after face recognition results) ───
    st.markdown("<hr style='border: 1px solid #a020f0; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3>🐞 Debug</h3>", unsafe_allow_html=True)
    with st.expander("Debug Data (JSON)", expanded=False):
        # Merge KYC debug JSON and Face verification debug JSON under a single 'data' object
        debug_payload = st.session_state.get("kyc_debug_json", {"data": {}})
        if "data" not in debug_payload:
            debug_payload["data"] = {}
        face_dbg = st.session_state.get("face_debug_json")
        if isinstance(face_dbg, dict) and "data" in face_dbg and "face_verification" in face_dbg["data"]:
            debug_payload["data"]["face_verification"] = face_dbg["data"]["face_verification"]
        st.json(debug_payload)
    # ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

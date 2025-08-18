import streamlit as st
import requests
import json
import re
import os
import time
import mimetypes
from textwrap import dedent
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import streamlit.components.v1 as components # <-- 1. IMPORTED COMPONENTS

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

def get_llm(model_choice):
    model_map = {
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "OpenAI": "openai/gpt-4o",
    }
    return ChatOpenAI(
        temperature=0.2,
        model_name=model_map[model_choice],
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=2000,
    )

def clean_address_mistral(raw_response, original_text=""):
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
                    "error": "No JSON block found",
                    "raw_output": raw_output,
                }
            )
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

# ── Table builder fixed using `dedent` to avoid markdown code-block rendering ──
def format_verification_table(results):
    """
    Builds a clean, card-style verification table similar to the AP Agent sample.
    Returns an HTML string (flush-left) ready for st.markdown(..., unsafe_allow_html=True).
    """
    if not results:
        return ""

    final_ok = bool(results.get("final_result", False))
    final_color = "#4CAF50" if final_ok else "#F44336"
    final_bg = "rgba(76,175,80,0.08)" if final_ok else "rgba(244,67,54,0.08)"
    final_text = "Verification Passed" if final_ok else "Verification Failed"

    # Count documents based on existing extracted_address_* keys
    doc_count = len([k for k in results.keys() if k.startswith("extracted_address_")])

    # --- Card header (status + key scores) ---
    header_html = f"""
<div style="background:#ffffff; border-radius:20px; box-shadow:0 2px 16px rgba(25,39,64,0.08); padding:22px 22px 18px 22px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
    <div style="border:2px solid {final_color}; background:{final_bg}; color:{final_color};
                font-weight:800; padding:8px 14px; border-radius:12px; font-size:18px;">
      {final_text}
    </div>
    <div style="color:#24345C; font-weight:600;">Address Consistency:
      <span style="color:{final_color}; font-weight:800;">
        {int(results.get("address_consistency_score", 0) * 100)}%
      </span>
    </div>
    <div style="color:#24345C; font-weight:600;">Name Consistency:
      <span style="color:{final_color}; font-weight:800;">
        {int(results.get("name_consistency_score", 0) * 100)}%
      </span>
    </div>
    <div style="color:#24345C; font-weight:600;">Overall Consistency:
      <span style="color:{final_color}; font-weight:800;">
        {int(results.get("document_consistency_score", 0) * 100)}%
      </span>
    </div>
    <div style="color:#24345C; font-weight:600;">Avg Authenticity:
      <span style="color:{final_color}; font-weight:800;">
        {int(results.get("average_authenticity_score", 0) * 100)}%
      </span>
    </div>
  </div>
</div>
"""

    # --- Table (styled like AP Agent sample) ---
    table_head = """
<table style="width:100%; border-collapse:collapse; background:#ffffff; border-radius:16px; overflow:hidden;
             box-shadow:0 2px 16px rgba(25,39,64,0.06);">
  <thead>
    <tr style="background:#f7f7fa;">
      <th style="text-align:left; padding:12px 14px; border-bottom:1px solid #ebeef5; width:6%;">Doc</th>
      <th style="text-align:left; padding:12px 14px; border-bottom:1px solid #ebeef5; width:28%;">Address</th>
      <th style="text-align:left; padding:12px 14px; border-bottom:1px solid #ebeef5; width:26%;">Full Name</th>
      <th style="text-align:center; padding:12px 14px; border-bottom:1px solid #ebeef5; width:10%;">Similarity %</th>
      <th style="text-align:center; padding:12px 14px; border-bottom:1px solid #ebeef5; width:10%;">Address Match</th>
      <th style="text-align:center; padding:12px 14px; border-bottom:1px solid #ebeef5; width:10%;">Google Maps</th>
      <th style="text-align:center; padding:12px 14px; border-bottom:1px solid #ebeef5; width:10%;">Authenticity %</th>
    </tr>
  </thead>
  <tbody>
"""

    def pill(text, ok):
        return f"""
<span style="
  display:inline-block; padding:4px 10px; border-radius:999px;
  font-weight:700; font-size:12px;
  color:{'#1B5E20' if ok else '#B71C1C'};
  background:{'rgba(76,175,80,0.12)' if ok else 'rgba(244,67,54,0.12)'};
  border:1px solid {'#C8E6C9' if ok else '#FFCDD2'};
">
  {text}
</span>
"""

    rows_html = ""
    for idx in range(doc_count):
        address = results.get(f"extracted_address_{idx+1}", "Not provided")
        name = results.get(f"extracted_name_{idx+1}", "Not provided")
        sim_pct = int(round(results.get(f"similarity_to_expected_{idx+1}", 0) * 100))
        match = bool(results.get(f"address_match_{idx+1}", False))
        maps_ok = bool(results.get(f"google_maps_verified_{idx+1}", False))
        auth_pct = int(round(results.get(f"authenticity_score_{idx+1}", 0) * 100))

        rows_html += f"""
    <tr>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; color:#24345C; font-weight:700;">{idx+1}</td>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; word-break:break-word; color:#324a6d;">{address}</td>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; word-break:break-word; color:#324a6d;">{name}</td>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; text-align:center; color:#24345C; font-weight:700;">{sim_pct}%</td>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; text-align:center;">{pill('Yes' if match else 'No', match)}</td>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; text-align:center;">{pill('Yes' if maps_ok else 'No', maps_ok)}</td>
      <td style="padding:10px 14px; border-bottom:1px solid #f0f2f7; text-align:center; color:#24345C; font-weight:700;">{auth_pct}%</td>
    </tr>
"""

    table_html = table_head + rows_html + """
  </tbody>
</table>
"""

    # Wrap in a card container to match the AP Agent “block-card” feel
    outer = f"""
<div style="background:transparent;">
  {header_html}
  <div style="margin-top:12px;">
    {table_html}
  </div>
</div>
"""
    # Ensure no leading indentation to avoid code-block rendering in Markdown
    return dedent(outer)

def kyc_multi_verify(files, expected_address, model_choice, consistency_threshold):
    if not files or len(files) < 2:
        return "❌ Please upload at least two documents.", "", {}

    try:
        results = {}
        kyc_fields = {}
        addresses = []
        names = []
        authenticity_scores = []

        for idx, file in enumerate(files):
            text = extract_text_from_file(file)
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
        results["name_consistency_score"] = (
            semantic_match(names[0], names[1])[0]
            if names[0] != "Not provided" and names[1] != "Not provided"
            else 0
        )
        results["document_consistency_score"] = round(address_consistency_score, 3)
        results["documents_consistent"] = address_consistent
        results["average_authenticity_score"] = round(avg_authenticity_score, 3)

        results["final_result"] = all(
            [results[f"address_match_{i+1}"] and results[f"google_maps_verified_{i+1}"] for i in range(len(files))]
        ) and (address_consistency_score >= consistency_threshold)

        status = (
            f"✅ <b style='color:green;'>Verification Passed</b><br>"
            f"Address Consistency Score: <b>{int(round(address_consistency_score * 100))}%</b><br>"
            f"Name Consistency Score: <b>{int(round(results['name_consistency_score'] * 100))}%</b><br>"
            f"Overall Consistency Score: <b>{int(round(results['document_consistency_score'] * 100))}%</b><br>"
            f"Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
            if results["final_result"]
            else f"❌ <b style='color:red;'>Verification Failed</b><br>"
                 f"Address Consistency Score: <b>{int(round(address_consistency_score * 100))}%</b><br>"
                 f"Name Consistency Score: <b>{int(round(results['name_consistency_score'] * 100))}%</b><br>"
                 f"Overall Consistency Score: <b>{int(round(results['document_consistency_score'] * 100))}%</b><br>"
                 f"Average Authenticity Score: <b>{int(round(avg_authenticity_score * 100))}%</b>"
        )
        return status, format_verification_table(results), kyc_fields

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

/* Markdown outputs inside app */
.stMarkdown, .stText, .stCodeBlock { color: #ffffff !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<h1>EZOFIS KYC Agent</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<span class='purple-circle'>1</span> <b>Upload Documents (2 or more)</b>", unsafe_allow_html=True)
        file_inputs = st.file_uploader(
            "Documents", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"]
        )

    with col2:
        st.markdown("<span class='purple-circle'>2</span> <b>Enter Expected Address</b>", unsafe_allow_html=True)
        expected_address = st.text_input("Expected Address", placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2")

        st.markdown("<span class='purple-circle'>3</span> <b>Select LLM Provider</b>", unsafe_allow_html=True)
        model_choice = st.selectbox("LLM Provider", ["Mistral", "OpenAI"], index=0)

        st.markdown("<span class='purple-circle'>4</span> <b>Set Consistency Threshold</b>", unsafe_allow_html=True)
        consistency_threshold = st.slider("Consistency Threshold", min_value=0.5, max_value=1.0, value=0.82, step=0.01)

    verify_btn = st.button("🔍 Verify Now")

    st.markdown("<span class='purple-circle'>5</span> <b>KYC Verification Status</b>", unsafe_allow_html=True)
    status_placeholder = st.empty()
    
    # This container will hold the results table and JSON data
    results_container = st.container()

    if verify_btn:
        if file_inputs and len(file_inputs) >= 2 and expected_address and model_choice and consistency_threshold:
            with st.spinner("Verifying... This may take a moment."):
                status, output_html, document_info_json = kyc_multi_verify(
                    file_inputs, expected_address, model_choice, consistency_threshold
                )
                status_placeholder.markdown(status, unsafe_allow_html=True)

                # Use the results_container to display the output
                with results_container:
                    st.markdown("<span class='purple-circle'>6</span> <b>KYC Verification Details</b>", unsafe_allow_html=True)
                    with st.expander("View Full Verification Details", expanded=True): # <-- 3. EXPANDED BY DEFAULT
                        
                        # <-- 2. USE COMPONENTS.HTML FOR THE TABLE
                        # This bypasses the markdown parser, rendering the HTML directly and correctly.
                        # The height parameter helps contain the table within a scrollable view if it gets long.
                        if output_html:
                            components.html(output_html, height=500, scrolling=True)
                        else:
                            st.info("No verification details to display.")
                        
                        st.markdown("### Extracted Document Details (JSON)")
                        st.json(document_info_json)
        else:
            st.error(
                "Please provide all required inputs: at least two documents, expected address, and an LLM provider."
            )

if __name__ == "__main__":
    main()
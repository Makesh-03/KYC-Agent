import os
import re
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
        max_tokens=200,
    )

def clean_address_mistral(raw_response):
    flattened = raw_response.replace("\n", ", ").replace("  ", " ")
    match = re.search(
        r"\d{1,5}[\w\s.,'-]+?,\s*\w+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z][ ]?\d[A-Z]\d",
        flattened,
        re.IGNORECASE,
    )
    return match.group(0).strip() if match else flattened.strip()

def extract_address_with_llm(text, model_choice):
    if model_choice == "Mistral":
        template = (
            "ONLY return the full Canadian mailing address from the text below. "
            "No notes or explanations. Include number, street, city, province, and postal code.\n\n"
            "Text: {document_text}\n\nAddress:"
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

    return clean_address_mistral(raw) if model_choice == "Mistral" else raw

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
        return "❌ Verification Failed: Please upload both documents.", {}

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

        passed = all([match1, match2, verified1, verified2, consistent])
        if passed:
            status = f"✅ <b style='color:green;'>Verification Passed</b><br>Consistency Score: <b>{percent_score}%</b>"
        else:
            status = f"❌ <b style='color:red;'>Verification Failed</b><br>Consistency Score: <b>{percent_score}%</b>"

        return status, {
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

    except Exception as e:
        return f"❌ <b style='color:red;'>Error:</b> {str(e)}", {}

# UI
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
            gr.Markdown("<span class='purple-circle'>1</span> **Upload Document 1 (e.g., License)**")
            file_input_1 = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg"], label="Document 1")
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>2</span> **Enter Expected Address**")
            expected_address = gr.Textbox(
                label="Expected Address",
                placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>3</span> **Upload Document 2 (e.g., Void Cheque)**")
            file_input_2 = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg"], label="Document 2")
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>4</span> **Select LLM Provider**")
            model_choice = gr.Dropdown(
                choices=["Mistral", "OpenAI"], value="Mistral", label="LLM Provider"
            )
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

    verify_btn.click(
        fn=kyc_dual_verify,
        inputs=[file_input_1, file_input_2, expected_address, model_choice],
        outputs=[status_html, output_json]
    )

if __name__ == "__main__":
    iface.launch()

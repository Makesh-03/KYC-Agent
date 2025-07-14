import os
import mimetypes
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Model and API Configuration ---

# Load sentence transformer model for semantic similarity
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM for intelligent address extraction
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)

# Get Canada Post API key from environment
CANADA_POST_API_KEY = os.getenv("CANADA_POST_API_KEY")

# --- Core Functions ---

def extract_text_from_file(file_path):
    """Extracts raw text from a PDF or image file."""
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type == "application/pdf":
            elements = partition_pdf(file_path)
        elif mime_type and mime_type.startswith("image/"):
            elements = partition_image(filename=file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or image.")
        return "\n".join([str(e) for e in elements])
    except Exception as e:
        if "OCRAgent" in str(e):
            raise RuntimeError(
                "OCR processing failed. Please ensure Tesseract is installed and "
                "that the 'unstructured[local-inference]' package is installed."
            )
        raise e

def extract_address_with_llm(text):
    """Uses an LLM to intelligently extract the address from the text with a few-shot prompt."""
    prompt = PromptTemplate(
        template=(
            "Please extract the full Canadian mailing address from the following text. "
            "The address should include the street, city, province, and postal code. "
            "Here is an example of how to do it:\n\n"
            "--- Example ---\n"
            "Text: 'Driver Licence /Permis oes 4a.nonr M981 209050 ‘4a, tss/Dé1 2024/06/28 3. DOB/DDN 4b.Exp 2028/12/09 1. MANICKAM THAMARAI 2. MAKESH KARTHIK 8.2 THORBURN ROAD ST JOHNS, NL A1B 3L7 15. Sexisexe M 46, HgtTaille 167 48. EyesYeux BRO BRN 9. Class/Classe 05 9a oi'\n"
            "Address: 2 THORBURN ROAD ST JOHNS, NL A1B 3L7\n"
            "--- End Example ---\n\n"
            "Now, please extract the address from the following text. If no address is found, return an empty string.\n\n"
            "Text: {document_text}\n\nAddress:"
        ),
        input_variables=["document_text"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"document_text": text})
    return result["text"].strip()

def semantic_match(extracted, expected, threshold=0.85):
    """Calculates the semantic similarity between two addresses."""
    embeddings = similarity_model.encode([extracted, expected], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return sim.item(), sim.item() >= threshold

def verify_with_canada_post(address):
    """Verifies an address using the Canada Post AddressComplete API."""
    if not CANADA_POST_API_KEY:
        return {"error": "CANADA_POST_API_KEY secret not set in Hugging Face Space settings."}
    try:
        url = "https://ws1.postescanada-canadapost.ca/AddressComplete/Interactive/Find/v2.10/json3.ws"
        response = requests.get(
            url, params={"Key": CANADA_POST_API_KEY, "Text": address, "Country": "CAN"}
        )
        data = response.json()
        return len(data.get("Items", [])) > 0
    except Exception as e:
        print(f"Canada Post API error: {e}")
        return False

# --- Main KYC Verification Workflow ---

def kyc_verify(file, expected_address):
    """The main function that orchestrates the entire KYC verification process."""
    if file is None:
        return {"error": "Please upload a document to verify."}
    try:
        # 1. Extract text from the uploaded document
        text = extract_text_from_file(file.name)
        if not text:
            return {"error": "Could not extract any text from the document."}

        # 2. Use the LLM to intelligently find the address
        extracted_address = extract_address_with_llm(text)
        if not extracted_address:
            return {"error": "Could not find a valid address in the document."}

        # 3. Perform semantic and Canada Post verifications
        sim_score, sem_ok = semantic_match(extracted_address, expected_address)
        cp_ok = verify_with_canada_post(extracted_address)

        # 4. Compile and return the final result
        return {
            "extracted_address": extracted_address,
            "semantic_similarity": round(sim_score, 3),
            "address_match": sem_ok,
            "canada_post_verified": cp_ok,
            "final_result": sem_ok and (cp_ok if cp_ok is not None else True),
        }
    except Exception as e:
        return {"error": str(e)}

# --- Custom CSS for Styling ---
custom_css = """
h1, .gr-textbox label, .gr-file label {
    font-size: 20px !important;
    font-weight: bold;
}

.purple-circle {
    display: inline-block;
    background-color: #6a0dad;
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    text-align: center;
    line-height: 28px;
    margin-right: 8px;
    font-weight: bold;
}

.purple-button button {
    background-color: #6a0dad !important;
    color: white !important;
    font-weight: bold;
    font-size: 16px;
    padding: 10px 16px;
}
"""

# --- Gradio Interface with Layout and Styling ---
with gr.Blocks(css=custom_css, title="🇨🇦 Intelligent KYC Document Verifier") as iface:
    gr.Markdown(
        """
        # 🇨🇦 Intelligent KYC Document Verifier  
        Upload a Canadian document and this AI agent will intelligently find and verify the address using:
        - 🧠 LLM for extraction  
        - 🔍 Semantic similarity  
        - 📫 Canada Post API for validation  
        
        **⚠️ Privacy Warning:** This is a public demo. Do not upload real, sensitive documents.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>1</span> **Upload Document**")
            file_input = gr.File(
                label="Upload Passport or Bill (PDF or Image)",
                file_types=["pdf", "image"]
            )
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>2</span> **Enter Expected Address**")
            expected_address = gr.Textbox(
                label="Expected Address",
                placeholder="e.g., 123 Main St, Toronto, ON, M5V 2N2"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<span class='purple-circle'>3</span> **KYC Verification Results**")
            output_json = gr.JSON(label="Verification Output")

    with gr.Row():
        verify_btn = gr.Button("🔍 Verify Now", elem_classes="purple-button")

    verify_btn.click(
        fn=kyc_verify,
        inputs=[file_input, expected_address],
        outputs=output_json
    )

if __name__ == "__main__":
    iface.launch()

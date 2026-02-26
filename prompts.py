"""Prompt templates for each benchmark type, with model-specific overrides."""

# Default prompts (fallback for unknown models like MinerU)
PROMPTS: dict[str, str | None] = {
    # OmniDocBench official prompt — standard across 21 inference scripts
    # (Qwen2VL_img2md.py, gpt_4o_inf.py, gemini25_img2md.py, InternVL2, etc.)
    "document_parse": (
        "You are an AI assistant specialized in converting PDF images to Markdown format.\n"
        "Please follow these instructions for the conversion:\n\n"
        "1. Text Processing:\n"
        "- Accurately recognize all text content in the PDF image without guessing or inferring.\n"
        "- Convert the recognized text into Markdown format.\n"
        "- Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n"
        "2. Mathematical Formula Processing:\n"
        "- Convert all mathematical formulas to LaTeX format.\n"
        "- Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)\n"
        "- Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]\n\n"
        "3. Table Processing:\n"
        "- Convert tables to HTML format.\n"
        "- Wrap the entire table with <table> and </table>.\n\n"
        "4. Figure Handling:\n"
        "- Ignore figures content in the PDF image. Do not attempt to describe or convert images.\n\n"
        "5. Output Format:\n"
        "- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.\n"
        "- For complex layouts, try to maintain the original document's structure and format as closely as possible.\n\n"
        "Please strictly follow these guidelines to ensure accuracy and consistency in the conversion."
    ),
    "document_parse_dp": (
        "Parse this document page. For each element, output its type and content "
        "in markdown format. Preserve the document structure and reading order."
    ),
    "text_recognition": None,  # OCRBench uses its own question field
    "formula_recognition": (
        "Convert the mathematical formula in this image to LaTeX. "
        "Output ONLY the LaTeX code."
    ),
    "table_to_html": (
        "Convert the table in this image to HTML. Output a complete <table> "
        "element with proper <thead>, <tbody>, <tr>, <td>, <th> tags. "
        "Include colspan/rowspan."
    ),
    "kie_extraction": (
        "Extract key information from this document as JSON with fields: "
        "date, doc_no_receipt_no, seller_name, seller_address, seller_gst_id, "
        "seller_phone, total_amount, total_tax. Use null for missing fields."
    ),
    "handwritten": (
        "Transcribe the handwritten text in this image. "
        "Output ONLY the transcribed text."
    ),
}

# Model-specific prompt overrides
# These override the default prompts when the model is known.
# GLM-OCR (standalone VLM): uses "Text Recognition:", "Table Recognition:", etc.
# PaddleOCR-VL (standalone VLM): uses "OCR:", "Table Recognition:", etc.
# DeepSeek-OCR2: uses official OmniDocBench prompt (default) for document_parse
# Pipeline models: document_parse/dp set to None (SDK handles prompting internally)
MODEL_PROMPTS: dict[str, dict[str, str | None]] = {
    "GLM-OCR": {
        "document_parse": "Convert the document to markdown.",
        "document_parse_dp": "Text Recognition:",
        "text_recognition": None,  # OCRBench uses its own question
        "formula_recognition": "Formula Recognition:",
        "table_to_html": "Table Recognition:",
        "kie_extraction": None,  # use default JSON extraction prompt
        "handwritten": "Text Recognition:",
    },
    "GLM-OCR-Pipeline": {
        "document_parse": None,  # SDK handles prompting internally
        "document_parse_dp": None,
        "text_recognition": None,
        "formula_recognition": "Formula Recognition:",
        "table_to_html": "Table Recognition:",
        "kie_extraction": None,
        "handwritten": "Text Recognition:",
    },
    "PaddleOCR-VL": {
        "document_parse": "Convert the document to markdown.",
        "document_parse_dp": "OCR:",
        "text_recognition": None,  # OCRBench uses its own question
        "formula_recognition": "Formula Recognition:",
        "table_to_html": "Table Recognition:",
        "kie_extraction": None,  # use default JSON extraction prompt
        "handwritten": "OCR:",
    },
    "PaddleOCR-VL-Pipeline": {
        "document_parse": None,  # SDK handles prompting internally
        "document_parse_dp": None,
        "text_recognition": None,
        "formula_recognition": "Formula Recognition:",
        "table_to_html": "Table Recognition:",
        "kie_extraction": None,
        "handwritten": "OCR:",
    },
    "DeepSeek-OCR2": {
        "document_parse": "Convert the document to markdown.",
        "document_parse_dp": "Convert the document to markdown.",
        "text_recognition": None,  # OCRBench uses its own question
        "formula_recognition": "Free OCR.",
        "table_to_html": "Free OCR.",
        "kie_extraction": None,  # use default JSON extraction prompt
        "handwritten": "Free OCR.",
    },
}


def get_prompt(prompt_key: str, sample_question: str | None = None, model_name: str | None = None) -> str:
    """Get the prompt for a benchmark, optionally model-specific.

    For text_recognition (OCRBench), uses the sample's own question field.
    For known models, uses model-specific prompt overrides.
    """
    if prompt_key == "text_recognition":
        return sample_question or "Read the text in this image."

    # Try model-specific prompt first
    if model_name and model_name in MODEL_PROMPTS:
        prompt = MODEL_PROMPTS[model_name].get(prompt_key)
        if prompt is not None:
            return prompt

    # Fall back to default
    prompt = PROMPTS.get(prompt_key)
    if prompt is None:
        raise ValueError(f"Unknown prompt key: {prompt_key}")
    return prompt

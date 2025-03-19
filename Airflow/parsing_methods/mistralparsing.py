import json
import base64
import requests
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("Mistral_API_KEY")
client = Mistral(api_key=api_key)

# Path configuration
OUTPUT_ROOT_DIR = Path("ocr_output")

# Ensure directories exist
OUTPUT_ROOT_DIR.mkdir(exist_ok=True)

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![Image](data:image/jpeg;base64,{base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def process_pdf(pdf_path):
    """Process a PDF file using Mistral OCR API and generate markdown with embedded images"""
    pdf_base = Path(pdf_path).stem
    print(f"Processing {pdf_path} using Mistral OCR...")

    # Set up output directory
    output_dir = OUTPUT_ROOT_DIR / f"extracted_{pdf_base}"
    output_dir.mkdir(exist_ok=True)

    # Read the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Encode PDF to base64 for API submission
    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

    # Process the PDF using Mistral OCR API
    try:
        # For simplicity, we're assuming the PDF is available online
        # In a real implementation, you might need to upload the PDF first
        # For now, we'll use a direct approach with the file itself
        ocr_response = client.ocr.process(
            document=pdf_path,
            model="mistral-ocr-latest",
            include_image_base64=True
        )

        # OCR -> Markdown with embedded images
        updated_markdown_pages = []

        for page in ocr_response.pages:
            updated_markdown = page.markdown
            for image_obj in page.images:
                base64_str = image_obj.image_base64
                if base64_str.startswith("data:"):
                    base64_str = base64_str.split(",", 1)[1]
                
                # Replace the image reference with an embedded base64 image
                updated_markdown = updated_markdown.replace(
                    f"![{image_obj.id}]({image_obj.id})",
                    f"![Image](data:image/jpeg;base64,{base64_str})"
                )
            updated_markdown_pages.append(updated_markdown)

        final_markdown = "\n\n".join(updated_markdown_pages)
        output_markdown_path = output_dir / "output.md"
        with open(output_markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(final_markdown)
        print(f"Markdown with embedded images generated in {output_markdown_path}")
        return output_markdown_path
    
    except Exception as e:
        print(f"Error processing PDF with Mistral OCR: {str(e)}")
        raise

if __name__ == "__main__":
    # For testing purposes
    sample_pdf = "sample.pdf"
    process_pdf(sample_pdf)

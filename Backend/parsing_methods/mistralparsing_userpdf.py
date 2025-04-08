import json
import base64
import shutil
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse

# The only requirement for this script is to have a Mistral API Key.
# You can get a free API Key at: https://console.mistral.ai/api-keys
# You can put the api key in the .env file (see the README.md for more information)
# or you can put it directly in the script below.

import os
from dotenv import load_dotenv

load_dotenv(override=True)

try:    
    # Option 1: Load API key from .env file
    api_key = os.getenv("Mistral_API_KEY")
    if not api_key:
        raise ValueError("Mistral_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error loading API key: {str(e)}")
    exit(1)
    

print(f"Loaded API Key: {api_key[:4]}...")
client = Mistral(api_key=api_key)

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    This converts base64 encoded images directly in the markdown...
    And replaces them with links to external images, so the markdown is more readable and organized.
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Part of the response from the Mistral API, which is an OCRResponse object...
    And returns a single string with the combined markdown of all the pages of the PDF.
    """
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)

def process_pdf(pdf_path: Path, output_dir: Path = None):
    """Processes a single PDF file, embeds images into the markdown,
    and eliminates the need for a separate JSON file."""

    # PDF base name
    pdf_base = pdf_path.stem
    print(f"Processing {pdf_path.name} ...")

    # Output directory (use provided output_dir if available)
    if output_dir:
        output_dir = Path(output_dir)

    # Define output path upfront so it's available in both success and error cases
    output_markdown_path = output_dir / "output.md"

    try:
        # PDF -> OCR
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_path.name,
                "content": pdf_bytes,
            },
            purpose="ocr"
        )

        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

        # Embed images directly into markdown
        markdown_content = ""
        for page in ocr_response.pages:
            for image_obj in page.images:
                # base64 to data URI
                base64_str = image_obj.image_base64
                if base64_str.startswith("data:"):
                    # Already a data URI, no need to modify
                    image_uri = base64_str
                else:
                    # Create data URI from base64 string
                    image_uri = f"data:image/png;base64,{base64_str}"

                # Update markdown with embedded image data URI
                page.markdown = page.markdown.replace(
                    f"![{image_obj.id}]({image_obj.id})",
                    f"![{image_obj.id}]({image_uri})"
                )
            markdown_content += page.markdown + "\n\n"  

        # Save markdown with embedded images
        with open(output_markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        print(f"Markdown with embedded images generated in {output_markdown_path}")
        
        # Return the path string
        return str(output_markdown_path)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Create error markdown
        with open(output_markdown_path, "w", encoding="utf-8") as f:
            f.write(f"# Processing Error\n\nFailed to process {pdf_path}\n\nError: {str(e)}")
        
        print(f"Error markdown saved to {output_markdown_path}")
        return str(output_markdown_path)
    
if __name__ == "__main__":
    process_pdf(Path(""))
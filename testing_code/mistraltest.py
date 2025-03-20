import base64
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("Mistral_API_KEY")
client = Mistral(api_key=api_key)

# Path configuration
PDF_URL = "https://arxiv.org/pdf/2201.04234"
OUTPUT_ROOT_DIR = Path("ocr_output")
OUTPUT_ROOT_DIR.mkdir(exist_ok=True)

def process_pdf(pdf_url: str):
    pdf_base = "extracted_pdf"
    print(f"Processing {pdf_url} ...")

    output_dir = OUTPUT_ROOT_DIR / pdf_base
    output_dir.mkdir(exist_ok=True)

    try:
        # Call Mistral OCR API
        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=pdf_url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

        # Process the OCR response and create markdown with embedded images
        markdown_pages = []

        for page in ocr_response.pages:
            # Start with the page's markdown content
            page_markdown = page.markdown
            
            # Replace image references with embedded base64 images
            for image_obj in page.images:
                base64_str = image_obj.image_base64
                if not base64_str.startswith("data:"):
                    # Add proper data URI prefix if not present
                    img_type = "jpeg"  # Default image type
                    if image_obj.id.lower().endswith(".png"):
                        img_type = "png"
                    elif image_obj.id.lower().endswith((".jpg", ".jpeg")):
                        img_type = "jpeg"
                    base64_str = f"data:image/{img_type};base64,{base64_str}"
                
                # Replace the original image reference with embedded base64 image
                page_markdown = page_markdown.replace(
                    f"![{image_obj.id}]({image_obj.id})",
                    f"![Image {image_obj.id}]({base64_str})"
                )
            
            markdown_pages.append(page_markdown)

        # Combine all pages into one markdown document
        final_markdown = "\n\n".join(markdown_pages)
        
        # Save just the markdown with embedded images
        output_markdown_path = output_dir / "output.md"
        with open(output_markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(final_markdown)
        print(f"Markdown with embedded images generated in {output_markdown_path}")
        return output_markdown_path
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Create a fallback markdown file
        output_markdown_path = output_dir / "output.md"
        with open(output_markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(f"# Processing Error\n\nFailed to process {pdf_url}\n\nError: {str(e)}")
        print(f"Error markdown saved to {output_markdown_path}")
        return output_markdown_path


# Run the PDF processing function
process_pdf(PDF_URL)
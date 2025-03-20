import os
import time
import requests
from pathlib import Path
from mistralai import Mistral,DocumentURLChunk
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("Mistral_API_KEY")
client = Mistral(api_key=api_key)

def process_pdf(pdf_url, output_dir=None):
    """
    Process a PDF using Mistral OCR with direct URL.
    
    Args:
        pdf_url: URL to the PDF file
        output_dir: Custom output directory (optional)
        
    Returns:
        Path to the generated markdown file
    """
    # Extract filename from URL for base name
    url_parts = pdf_url.split('/')
    pdf_base = url_parts[-1].split('.')[0] if '.' in url_parts[-1] else url_parts[-1]
    print(f"Processing PDF from URL: {pdf_url}")

    # Set up output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        # Create a directory based on the PDF base name
        output_dir = Path("./ocr_output") / pdf_base
    
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    try:
        # Add a delay between API calls to avoid rate limiting
        time.sleep(3)
        
        # Test URL accessibility before calling Mistral
        try:
            response = requests.head(pdf_url, timeout=5)
            if response.status_code != 200:
                print(f"Warning: URL returned status {response.status_code}")
        except Exception as e:
            print(f"Warning: Error testing URL: {e}")
        
        # Process using URL
        print(f"Calling Mistral OCR API with URL: {pdf_url}")
        ocr_response = client.ocr.process(  # Use process() instead of create()
            document=DocumentURLChunk(document_url=pdf_url),  # Use document_url parameter
            model="mistral-ocr-latest",
            include_image_base64=True  # Request images as base64
        )
        
        # Process OCR response
        print(f"Processing OCR response")
        if hasattr(ocr_response, 'pages'):
            pages = ocr_response.pages
        elif hasattr(ocr_response, 'result') and hasattr(ocr_response.result, 'pages'):
            pages = ocr_response.result.pages
        else:
            print(f"Unexpected response structure: {dir(ocr_response)}")
            raise ValueError("Could not find pages in OCR response")
            
        print(f"Found {len(pages)} pages in response")
        
        # Process each page into markdown with embedded images
        markdown_pages = []
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}")
            
            # Get the markdown content
            if hasattr(page, 'markdown'):
                page_content = page.markdown
            elif hasattr(page, 'text'):
                page_content = page.text
            else:
                page_content = f"# Page {i+1}\n\n[Content could not be extracted]"
            
            # Process images if available
            if hasattr(page, 'images') and page.images:
                for j, img in enumerate(page.images):
                    # Get image data
                    img_data = None
                    if hasattr(img, 'image_base64'):
                        img_data = img.image_base64
                    elif hasattr(img, 'base64'):
                        img_data = img.base64
                    
                    if img_data:
                        # Add data URI prefix if needed
                        if not img_data.startswith('data:'):
                            img_type = "jpeg"  # Default
                            if hasattr(img, 'id') and img.id.lower().endswith('.png'):
                                img_type = "png"
                            img_data = f"data:image/{img_type};base64,{img_data}"
                        
                        # Replace image reference with embedded base64
                        img_id = getattr(img, 'id', f"image_{i+1}_{j+1}")
                        page_content = page_content.replace(
                            f"![{img_id}]({img_id})",
                            f"![Image {i+1}-{j+1}]({img_data})"
                        )
            
            markdown_pages.append(page_content)
        
        # Combine into final markdown
        final_markdown = "\n\n".join(markdown_pages)
        
        # Save to file
        output_path = output_dir / "output.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        
        print(f"Markdown with embedded images generated at {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Create error markdown
        output_path = output_dir / "output.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Processing Error\n\nFailed to process {pdf_url}\n\nError: {str(e)}")
        
        print(f"Error markdown saved to {output_path}")
        return str(output_path)
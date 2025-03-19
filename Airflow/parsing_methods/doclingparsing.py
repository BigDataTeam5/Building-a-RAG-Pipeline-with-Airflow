import logging
import time
from pathlib import Path
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import boto3
import os

# AWS S3 Configuration
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'),
    aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY')
)
bucket_name = os.getenv('AWS_BUCKET_NAME')

# Constants
IMAGE_RESOLUTION_SCALE = 2.0

def upload_file_to_s3(file_path, object_name):
    """Uploads a file to S3."""
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        return f"Uploaded {file_path} to s3://{bucket_name}/{object_name}"
    except Exception as e:
        return f"Error uploading file {file_path}: {e}"

def main(pdf_path):
    """
    Convert a PDF document to markdown with embedded images using Docling.
    
    Args:
        pdf_path: Path to the PDF file to convert
    
    Returns:
        Path to the generated markdown file with embedded images
    """
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path(pdf_path)
    output_dir = Path(f"output/{input_doc_path.stem}")

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()
    try:
        # Convert the document
        conv_res = doc_converter.convert(input_doc_path)

        # Create local output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_filename = conv_res.input.file.stem

        # Save markdown with embedded images
        md_filename_embedded = output_dir / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(md_filename_embedded, image_mode=ImageRefMode.EMBEDDED)
        
        end_time = time.time() - start_time
        logging.info(f"Document converted and markdown with embedded images saved in {end_time:.2f} seconds.")
        
        return md_filename_embedded
    
    except Exception as e:
        logging.error(f"Error processing PDF with Docling: {str(e)}")
        raise

if __name__ == "__main__":
    # For testing purposes
    pdf_path = "doc.pdf"
    main(pdf_path)

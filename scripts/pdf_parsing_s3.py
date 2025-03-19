import os
import boto3
import docling
from mistralai import Mistral
from mistralai.models import DocumentURLChunk  # ✅ Import correct model for Mistral OCR
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Fetch AWS credentials from .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ✅ Debugging: Print values to ensure they are loaded correctly
print(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")
print(f"AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY_ID}")
print(f"AWS_SECRET_ACCESS_KEY: {'HIDDEN'}")
print(f"AWS_REGION: {AWS_REGION}")
print(f"MISTRAL_API_KEY: {'HIDDEN'}")

# ✅ Ensure no variable is None
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME, MISTRAL_API_KEY]):
    raise ValueError("⚠️ Missing required environment variables! Check your .env file.")

# ✅ Initialize AWS S3 client with explicit credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# ✅ Initialize Mistral OCR client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def list_s3_pdfs():
    """Lists all PDF files in the S3 bucket."""
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="nvidia_reports/")
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]

def parse_with_docling(pdf_bytes):
    """Extracts text using Docling for structured PDFs."""
    return docling.parse(pdf_bytes)

def parse_with_mistral_ocr(pdf_key):
    """Extracts text using Mistral OCR for scanned PDFs."""
    
    # ✅ Generate a secure, time-limited pre-signed URL (valid for 10 minutes)
    presigned_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": pdf_key},
        ExpiresIn=600  # 10 minutes
    )

    print(f"🔗 Generated Pre-Signed URL: {presigned_url}")

    # ✅ Use pre-signed URL instead of direct S3 link
    response = mistral_client.ocr.process(
        document=DocumentURLChunk(document_url=presigned_url),
        model="mistral-ocr-latest"
    )

    return "\n".join([page.markdown for page in response.pages])

def process_pdfs(parser_choice="Docling"):
    """Processes all PDFs from S3 using the selected parser."""
    pdf_files = list_s3_pdfs()

    if not pdf_files:
        print("⚠️ No PDFs found in S3!")
        return

    for pdf_key in pdf_files:
        print(f"📄 Processing {pdf_key} with {parser_choice}")

        if parser_choice == "Docling":
            pdf_bytes = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=pdf_key)['Body'].read()
            extracted_text = parse_with_docling(pdf_bytes)
        else:
            extracted_text = parse_with_mistral_ocr(pdf_key)  # ✅ Pass pdf_key, not URL

        # Store extracted text in S3
        extracted_key = f"parsed_reports/{pdf_key}.txt"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=extracted_key, Body=extracted_text.encode("utf-8"))
        print(f"✅ Extracted text uploaded to S3: {extracted_key}")

# Run the PDF extraction with user-selected method
if __name__ == "__main__":
    parser_choice = input("Choose a parser (Docling/Mistral): ").strip()
    if parser_choice not in ["Docling", "Mistral"]:
        print("⚠️ Invalid choice! Defaulting to Docling.")
        parser_choice = "Docling"
    process_pdfs(parser_choice)

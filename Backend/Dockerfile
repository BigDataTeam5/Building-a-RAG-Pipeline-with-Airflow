# Use an official Python runtime as a parent image
FROM python:3.10
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the requirements file and install dependencies
COPY requirements.txt .
 
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the FastAPI app code into the container
COPY . .
 
# Expose the port that FastAPI runs on
EXPOSE 8000
 
# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
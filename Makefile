.PHONY: install prerequisites backend frontend rag docker-up docker-down
# Install project dependencies using Poetry 
install:
	poetry install

lock:
	poetry lock
update:
	poetry update




	
backend:
	poetry run uvicorn backend.api:app --host 0.0.0.0 --port 8000
frontend:
	poetry run streamlit run .\frontend\app.py --server.port 8501
pinecone:
	poetry run python .\Rag_modelings\rag_pinecone.py

chromadb:
	poetry run python .\Rag_modelings\rag_chromadb.py

llm:
	poetry run python .\Backend\litellm_query_generator.py



#Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down
clean:
	docker-compose down --volumes --rmi all --remove-orphans

# testing code
fernet:
	poetry run python .\testing_code\fernet_key_generation_key.py

mistraltest:
	poetry run python .\testing_code\mistral_test.py

# parsing methods
docling:
	poetry run python .\parsing_methods\doclingparsing.py

mistral:
	poetry run python .\parsing_methods\mistral_userpdf.py



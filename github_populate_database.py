import os
import shutil
import requests
import time
import concurrent.futures
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

CHROMA_ROOT_PATH = "chroma"

def fetch_contents(api_url, token=None):
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    for attempt in range(5):  # Retry up to 5 times
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 403:  # Rate limit exceeded
            print("Rate limit exceeded, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            response.raise_for_status()
            return response
    
    # If we exhausted retries
    response.raise_for_status()

def process_directory(api_url, path="", token=None):
    file_contents = []
    while api_url:
        response = fetch_contents(api_url, token=token)
        contents = response.json()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_item = {executor.submit(process_item, item, path, token): item for item in contents}
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    file_contents.extend(result)
                except Exception as exc:
                    print(f'Item {item} generated an exception: {exc}')
        
        # GitHub API provides pagination info in headers
        api_url = response.links.get('next', {}).get('url')
    
    return file_contents

def process_item(item, path, token):
    file_contents = []
    if item['type'] == 'file':
        file_url = item['download_url']
        file_response = requests.get(file_url, headers={'Authorization': f'token {token}'})
        file_response.raise_for_status()
        file_contents.append({
            'name': os.path.join(path, item['name']),
            'content': file_response.text
        })
    elif item['type'] == 'dir':
        dir_url = item['url']
        file_contents.extend(process_directory(dir_url, path=os.path.join(path, item['name']), token=token))
    
    return file_contents

def scrape_github_repo(repo_url, token):
    repo_owner, repo_name = repo_url.rstrip('/').split('/')[-2:]
    api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents'
    return process_directory(api_url, token=token)

def clone_github_repo(repo_url, local_dir):
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    Repo.clone_from(repo_url, local_dir)

def read_local_repo(local_dir):
    file_contents = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_contents.append({
                    'name': os.path.relpath(file_path, local_dir),
                    'content': f.read()
                })
    return file_contents

def populate_chroma_from_data_github(repo_contents):
    chroma_path = CHROMA_ROOT_PATH

    # Ensure the Chroma directory exists
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)

    documents = []
    for file in repo_contents:
        doc = Document(page_content=file['content'], metadata={"source": file['name']})
        documents.append(doc)

    chunks = split_documents(documents)
    add_to_chroma(chroma_path, chunks)

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chroma_path, chunks):
    # Load the existing database.
    db = Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db._collection.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_ROOT_PATH):
        shutil.rmtree(CHROMA_ROOT_PATH)
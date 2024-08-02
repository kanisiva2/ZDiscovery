import os
import shutil
import psycopg2
from psycopg2 import sql
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

CHROMA_ROOT_PATH = "chroma"

# Database connection parameters
DB_HOST = "localhost"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASSWORD = "mypassword"

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

def create_table():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL
                )
            """)
            conn.commit()

def add_document_to_db(content):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (content) VALUES (%s)
            """, (content,))
            conn.commit()

def fetch_all_documents():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, content FROM documents")
            rows = cur.fetchall()
            return rows

def populate_chroma_from_db():
    documents = []
    for doc_id, content in fetch_all_documents():
        doc = Document(page_content=content, metadata={"source": str(doc_id)})
        documents.append(doc)
    
    chunks = split_documents(documents)
    add_to_chroma(CHROMA_ROOT_PATH, chunks)

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chroma_path, chunks):
    db = Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db._collection.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"{chunk.metadata['source']}:{i}"
    return chunks

def clear_database():
    if os.path.exists(CHROMA_ROOT_PATH):
        shutil.rmtree(CHROMA_ROOT_PATH)

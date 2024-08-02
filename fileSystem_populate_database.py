import argparse
import os
import shutil
import speech_recognition as sr
from pydub import AudioSegment
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

CHROMA_ROOT_PATH = "chroma"
DATA_ROOT_PATH = "data"

def populate_chroma_from_data(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, DATA_ROOT_PATH)
            subdir = os.path.dirname(relative_path)
            chroma_path = os.path.join(CHROMA_ROOT_PATH, subdir)

            # Ensure the corresponding Chroma subdirectory exists
            if not os.path.exists(chroma_path):
                os.makedirs(chroma_path)

            if file.endswith(".pdf"):
                documents = load_documents(root)
            elif file.endswith(".wav") or file.endswith(".mp3"):
                documents = process_audio(file_path)
            else:
                continue

            chunks = split_documents(documents)
            add_to_chroma(chroma_path, chunks)


def load_documents(file_path):
    document_loader = PyPDFDirectoryLoader(file_path)
    return document_loader.load()


def process_audio(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    audio.export("temp.wav", format="wav")

    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data)

    os.remove("temp.wav")

    # Create a Document object from the transcription text
    doc = Document(
        page_content=transcription,
        metadata={"source": file_path}
    )
    return [doc]


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chroma_path, chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
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
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_ROOT_PATH):
        shutil.rmtree(CHROMA_ROOT_PATH)


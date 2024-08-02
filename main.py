import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import streamlit as st
from get_embedding_function import get_embedding_function
from fileSystem_populate_database import populate_chroma_from_data
from github_populate_database import scrape_github_repo, populate_chroma_from_data_github
from database_populate_database import fetch_all_documents, populate_chroma_from_db

CHROMA_ROOT_PATH = "chroma"
DATA_ROOT_PATH = "/Users/kanisiva/Documents/VSCode/Other/Zillion/consolidate/data"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def rag_query(query):
    db = Chroma(persist_directory=CHROMA_ROOT_PATH, embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query, k=5)

    result_list = []
    for doc, _score in results:
        result_list.append({
            "file_path": doc.metadata.get("id", "Unknown"),
            "content": doc.page_content[:200] + "..."  # Display first 200 characters of content
    })

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"
    return result_list, formatted_response

def display_documents():
    st.subheader("All Documents")
    with st.expander("Show All Documents"):
        documents = fetch_all_documents()
        for doc_id, content in documents:
            st.write(f"Document ID: {doc_id}\nContent: {content}\n")

# Scraping page
def scraping_page():
    st.title("Data Scraping and Indexing")

    source_options = st.multiselect(
        "Select sources to scrape from:",
        ["GitHub", "File System", "PostgreSQL Database"]
    )

    if "GitHub" in source_options:
        repo_url = st.text_input("Enter the GitHub repository URL:", "")
        token = st.text_input("Enter your GitHub Token:", type="password")
        

    if "File System" in source_options:
        # Allow user to specify file directory to process
        data_directory = st.text_input("Enter the file system directory to process:", DATA_ROOT_PATH)

    if "PostgreSQL Database" in source_options:
        display_documents()

    if st.button("Scrape and Index"):
        if "GitHub" in source_options:
            if repo_url and token:
                with st.spinner("Scraping and indexing the GitHub repository..."):
                    repo_contents = scrape_github_repo(repo_url, token)
                    populate_chroma_from_data_github(repo_contents)
                st.write("GitHub repository scraped and indexed successfully!")
                print("GitHub repository scraped and indexed successfully!")
            else:
                st.write("Please provide the repository URL and token.")

        if "File System" in source_options:
            if os.path.exists(data_directory):
                with st.spinner(f"Processing file system directory: {data_directory}..."):
                    populate_chroma_from_data(data_directory)
                st.write("File system directory processed and indexed successfully!")
                print("File system directory processed and indexed successfully!")
            else:
                st.write("The specified directory does not exist.")

        if "PostgreSQL Database" in source_options:
            with st.spinner("Populating Chroma with PostgreSQL database content..."):
                populate_chroma_from_db()
            st.write("Chroma populated with database content successfully!")

# Search page
def search_page():
    st.title("Search Query")

    query = st.text_input("Enter your search query:")
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results, answer = rag_query(query)
            st.header(f"Answer for '{query}':")
            st.write(answer)
            st.header(f"Search Results for '{query}':")
            if results:
                for result in results:
                    st.write(f"File Path: {result['file_path']}")
                    st.write(f"Content Preview: {result['content']}")
                    st.markdown("---")
        else:
            st.write("Please enter a search query.")

# Main function to handle page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Scraping", "Search"])

    if page == "Scraping":
        scraping_page()
    elif page == "Search":
        search_page()

if __name__ == "__main__":
    main()

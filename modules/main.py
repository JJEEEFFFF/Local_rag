from modules.Text_processing import text_formatter,open_and_read_pdf,process_pages_with_spacy,sentence_chunk,process_chunks,join_and_clean_chunk
import os
import shutil
from docxtopdf import convert
from tqdm import tqdm
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

embedding_model=SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
nlp = English()


def batch_convert_to_pdf(source_dir: str, target_dir: str):
    """
    Converts DOCX files to PDF and copies existing PDF files to the target directory.

    Parameters:
        source_dir (str): Path to the directory containing the source files.
        target_dir (str): Path to the directory where PDF files will be saved.

    Returns:
        None
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate through files in the source directory
    for file_name in tqdm(os.listdir(source_dir), desc="Processing files"):
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        # Skip directories
        if os.path.isdir(source_path):
            continue

        # Check file extension
        if file_name.lower().endswith(".pdf"):
            # Copy PDF files directly
            shutil.copy(source_path, target_path)
        elif file_name.lower().endswith(".docx"):
            # Convert DOCX files to PDF
            pdf_file_name = os.path.splitext(file_name)[0] + ".pdf"
            pdf_target_path = os.path.join(target_dir, pdf_file_name)
            convert(source_path, pdf_target_path)
        else:
            # Skip unsupported file types
            print(f"Skipping unsupported file: {file_name}")


def process_pdf(directory_path:str,out_path:str):
    """
    Process each PDF file in the directory.

    Parameters:
    directory_path: str
        Path to the directory containing PDF files.

    Returns:
    list
        List of processed chunks for all files.
    """
    all_chunks = []  # Collect processed data for all files

    for file_name in tqdm(os.listdir(directory_path), desc="Processing files"):
        if not file_name.lower().endswith(".pdf"):
            continue  # Skip non-PDF files

        try:
            source_path = os.path.join(directory_path, file_name)
            text = text_formatter(source_path)
            read_pdf = open_and_read_pdf(text)
            spacy = process_pages_with_spacy(read_pdf)
            chunks_list = sentence_chunk(spacy, chunk_size=10)
            Chunks = process_chunks(chunks_list)
            
            # Save as JSON
            target_path = os.path.join(out_path, os.path.splitext(file_name)[0] + ".json")
            cleaned_data = pd.DataFrame(Chunks)
            cleaned_data.to_json(target_path, index=False)
            
            all_chunks.append(Chunks)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    return all_chunks  # Return processed data for all files

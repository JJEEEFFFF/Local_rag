import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer
import roman


embedding_model=SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
nlp = English()
nlp.add_pipe("sentencizer")

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text



def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number-6,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(".")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts


def process_pages_with_spacy(pages_and_texts: list[dict]) -> list[dict]:
    """
    Processes a list of dictionaries containing page text using spaCy.
    Splits the text into sentences and counts the sentences for each page.

    Parameters:
        page_data (list[dict]): A list of dictionaries, each containing a "text" key
                                with the text content of a page.

    Returns:
        list[dict]: The input list, with each dictionary updated to include
                    a "spacy_sentence_count" key with the number of sentences.
    """ 
    for item in tqdm(pages_and_texts):
    # Use SpaCy's NLP pipeline to split the text into sentences
        item["sentences"] = list(nlp(item["text"]).sents)
    
    # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    
    # Count the sentences 
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts
    

def split_list(input_list, slice_size):
    """
    Splits a list into smaller sublists of a specified size.
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def sentence_chunk(pages_and_texts: list[dict], chunk_size: int) -> list[dict]:
    """
    Loops through pages and texts, splitting sentences into chunks of a specified size.

    Parameters:
    - pages_and_texts (list[dict]): List of dictionaries, each containing a 'sentences' key.
    - chunk_size (int): Number of sentences per chunk.

    Returns:
    - list[dict]: The updated list of dictionaries with 'sentence_chunks' and 'num_chunks' added.
    """
    # Validate chunk_size
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    # Loop through each item in pages_and_texts
    for item in tqdm(pages_and_texts, desc="Chunking sentences"):
        # Check if the item contains valid 'sentences'
        if "sentences" not in item or not isinstance(item["sentences"], list):
            raise ValueError(f"Each item must contain a 'sentences' key with a list of sentences. Problematic item: {item}")
        
        # Split sentences into chunks and count them
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    
    return pages_and_texts
                    
def join_and_clean_chunk(sentence_chunk: list[str]) -> str:
    """
    Joins a list of sentences into a single string and performs basic cleaning.

    Args:
        sentence_chunk (list[str]): A list of sentences to be joined and cleaned.
    
    Returns:
        str: The joined and cleaned sentence chunk.
    """
    # Join sentences into a single string and replace multiple spaces with one
    joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
    
    # Add a space after full stops if followed by a capital letter
    cleaned_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
    
    return cleaned_chunk



def process_chunks(pages_and_texts: list[dict]) -> list[dict]:
    """
    Processes sentence chunks for each page and generates statistics for each chunk.

    Args:
        pages_and_texts (list[dict]): A list of dictionaries where each dictionary contains 
                                      'sentence_chunks' for each page.

    Returns:
        list[dict]: A list of dictionaries with chunk details like page number, sentence chunk, 
                    character count, word count, and token count.
    """
    pages_and_chunks = []

    for item in tqdm(pages_and_texts, desc="Processing sentence chunks"):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": join_and_clean_chunk(sentence_chunk),
                "chunk_char_count": len(join_and_clean_chunk(sentence_chunk)),
                "chunk_word_count": len(join_and_clean_chunk(sentence_chunk).split(" ")),
                "chunk_token_count": len(join_and_clean_chunk(sentence_chunk)) / 4  # Approximate token count
            }
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks





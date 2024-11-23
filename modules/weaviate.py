import weaviate
from weaviate.classes.init import Auth
import os
import pandas as pd
from tqdm.auto import tqdm
from weaviate.classes.config import Property, DataType
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5

# Best practice: store your credentials in environment variables
wcd_url = os.environ["WEAVIATE_URL"]
wcd_api_key = os.environ["WEAVIATE_API_KEY"]
openai_api_key = os.environ["OPENAI_APIKEY"]
cohere_api_key = os.environ["COHERE_APIKEY"]


with weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
    headers={
        'X-OpenAI-Api-key': openai_api_key,  # Replace with appropriate header key/value pair for the required API
        'X-Cohere-Api-Key': cohere_api_key
    }
) as client:
    client.connect()  # Use this context manager to ensure the connection is closed
    print(client.is_ready())


def create_collection(client: weaviate, collection_name: str) -> str:
    """
    Creates a collection in the Weaviate database with specific properties and configurations.

    Parameters:
        client (WeaviateClient): The Weaviate client instance to connect to the database.
        collection_name (str): The name of the collection to create.

    Returns:
        Collection: The created collection instance.
    """
    return client.collections.create(
        name=collection_name,
        properties=[
            Property(name="page_number", data_type=DataType.INT, skip_vectorization=True),
            Property(name="sentence_chunk", data_type=DataType.TEXT),
        ],
        vectorizer_config=wc.Configure.Vectorizer.text2vec_cohere(),
        generative_config=wc.Configure.Generative.openai(),
        inverted_index_config=wc.Configure.inverted_index(
            index_property_length=True
        )
    )
def batch_upload_to_weaviate(collection, Colloaction_data, batch_size=20, concurrent_requests=2
) -> dict:
    """
    Batch uploads data from a DataFrame to a Weaviate collection.

    Parameters:
        collection: The Weaviate collection instance to upload data to.
        df (pd.DataFrame): The DataFrame containing the data to upload.
        batch_size (int): The number of objects to upload per batch. Default is 50.
        concurrent_requests (int): The number of concurrent requests. Default is 2.

    Returns:
        dict: A mapping of page numbers to their corresponding UUIDs.
    """
    page_id_map = dict()

    with collection.batch.fixed_size(
        batch_size=batch_size, concurrent_requests=concurrent_requests
    ) as batch:
        for i, row in Colloaction_data.iterrows():
            page_number = row["page_number"]
            sentence_chunk = row["sentence_chunk"]

            # Generate properties and UUID for each page
            props = {
                "page_number": page_number,
                "sentence_chunk": sentence_chunk,
            }
            page_uuid = generate_uuid5(page_number)

            # Add page details to batch with the generated UUID
            batch.add_object(properties=props, uuid=page_uuid)

            # Store the UUID in the dictionary
            page_id_map[page_number] = page_uuid

    # Check for failed uploads
    if len(collection.batch.failed_objects) > 0 or len(collection.batch.failed_references) > 0:
        print("Failed Objects:", collection.batch.failed_objects[:5])
        print("Failed References:", collection.batch.failed_references[:5])

    return page_id_map


def process_file(file_name):
    """
    Processes a single file name by replacing spaces with underscores 
    and removing the extension.

    Parameters:
    - file_name (str): Name of the file to process.

    Returns:
    - str: Processed file name (without extension, spaces replaced by underscores).
    """
    # Split file name and extension
    name, ext = os.path.splitext(file_name)
    
    # Replace unwanted characters
    modified_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    
    return modified_name


def upload_collection(chunks_directory: str):
    """
    Processes all JSON files in the directory and uploads them to a Weaviate collection.

    Parameters:
    - chunks_directory (str): Path to the directory containing chunk files.

    Returns:
    - list: A list of results from collection creation and batch uploads.
    """
    weaviate_data = []
    
    for file_name in tqdm(os.listdir(chunks_directory), desc="Adding collections for each file to Weaviate"):
        if not file_name.lower().endswith(".json"):
            continue  # Skip non-JSON files

        try:
            # Construct the full path to the file
            file_path = os.path.join(chunks_directory, file_name)
            
            # Read the JSON file
            data = pd.read_json(file_path)
            
            # Process the file name
            processed_name = process_file(file_name)
            collection_name = client.collections.get(processed_name)
            # Create collection and batch upload
            collection_result = create_collection(client, processed_name)
            batch_upload_result = batch_upload_to_weaviate(collection_name, data)
            
            # Debugging: Check the types of the results
            print(f"Collection result type: {type(collection_result)}")
            print(f"Batch upload result type: {type(batch_upload_result)}")
            
            # Append results
            weaviate_data.append({
                "collection": collection_result,
                "batch_upload": batch_upload_result
            })
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    return weaviate_data

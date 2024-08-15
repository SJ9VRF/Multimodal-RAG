import fitz  # PyMuPDF
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.protobuf import json_format
from google.cloud.aiplatform_v1.types import PredictRequest, ExplanationSpec
import os

def extract_metadata(document_path):
    """
    Extracts metadata from a PDF document including text and images.
    Args:
    - document_path: The path to the PDF file.

    Returns:
    - A dictionary containing text and image metadata.
    """
    doc = fitz.open(document_path)
    metadata = {'text': [], 'images': []}
    for page in doc:
        text = page.get_text("text")
        metadata['text'].append(text)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]  # xref number
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            metadata['images'].append({
                'xref': xref,
                'image': image_data,
                'width': base_image['width'],
                'height': base_image['height']
            })
    doc.close()
    return metadata

def gemini_api_call(model_name, query):
    """
    Interacts with Google's Vertex AI to use the Gemini model for text or multimodal queries.
    Args:
    - model_name: The model name identifier on Vertex AI.
    - query: The query or prompt for the model.

    Returns:
    - The response from the model.
    """
    client = PredictionServiceClient()
    endpoint = client.endpoint_path(
        project=os.getenv('GOOGLE_CLOUD_PROJECT'),
        location='us-central1',  # Assuming the model is deployed in us-central1
        endpoint=model_name
    )

    # Assuming the query is plain text for simplicity. Adjust as necessary for multimodal inputs.
    instance = json_format.ParseDict({'content': query}, aiplatform.Example())
    response = client.predict(endpoint=endpoint, instances=[instance])

    # Process the response
    return response.predictions


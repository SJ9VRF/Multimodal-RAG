from utils import extract_metadata, gemini_api_call

def main():
    print("Extracting document metadata...")
    metadata = extract_metadata('data/documents/sample.pdf')
    
    print("Performing RAG with Gemini API...")
    response = gemini_api_call('gemini-1.5-pro', 'What is shown in the document?')
    
    print("Response:", response)

if __name__ == '__main__':
    main()

import os
from docx import Document

def extract_docx(input_path, output_path):
    print(f"Extracting text from {input_path}...")
    doc = Document(input_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write paragraph texts separated by newline
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))
        
    print(f"Saved extracted text to {output_path}")

if __name__ == "__main__":
    input_file = "data/raw/aman_internship_guide_2026.docx"
    output_file = "data/processed/aman_guide_raw.txt"
    extract_docx(input_file, output_file)

from flask import Flask, jsonify
import os
import pdfplumber
import docx
from PIL import Image
import pytesseract
import re
import threading

app = Flask(__name__)

# Configuration
app.config['RESUME_DIR'] = os.path.join('instance', 'resumes')
app.config['KNOWLEDGE_FILE'] = os.path.join('instance', 'knowledge.txt')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Create necessary directories
os.makedirs(app.config['RESUME_DIR'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['KNOWLEDGE_FILE']), exist_ok=True)

processing_lock = threading.Lock()

def clean_extra_spaces(content):
    """Clean whitespace while preserving line breaks"""
    cleaned = re.sub(r'\s+', ' ', content).strip()
    return '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())

def extract_content(file_path, ext):
    """Handle different file types"""
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)
                return clean_extra_spaces(text)[:5000]
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join(" ".join(para.text.split()) for para in doc.paragraphs)
            return clean_extra_spaces(text)[:3000]
        elif ext.lower() in ('.jpg', '.jpeg', '.png'):
            with Image.open(file_path) as img:
                text = pytesseract.image_to_string(img)
                cleaned_text = re.sub(r'\s+\n\s+', '\n', text)
                return clean_extra_spaces(cleaned_text)[:3000]
    except Exception as e:
        return f"Error processing file: {str(e)}"

def process_resumes():
    """Process resumes and update knowledge.txt"""
    processed = set()
    separator = "-" * 60
    
    with processing_lock:
        # Read existing processed files
        if os.path.exists(app.config['KNOWLEDGE_FILE']):
            with open(app.config['KNOWLEDGE_FILE'], 'r', encoding='utf-8') as f:
                content = f.read()
                processed_files = [line.split('\n')[0] for line in content.split(separator) if line.strip()]
                processed = set(processed_files)
        
        new_entries = []
        for filename in os.listdir(app.config['RESUME_DIR']):
            if filename in processed:
                continue
                
            file_path = os.path.join(app.config['RESUME_DIR'], filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in {'.pdf', '.docx', '.jpg', '.jpeg', '.png'}:
                continue
                
            content = extract_content(file_path, ext)
            new_entries.append(f"{separator}\n{filename}\n\n{content}\n")
        
        if new_entries:
            with open(app.config['KNOWLEDGE_FILE'], 'a', encoding='utf-8') as f:
                f.write('\n' + '\n'.join(new_entries))
                
        return len(new_entries)

@app.route('/Filter', methods=['POST'])
def filter_resumes():
    """Process resumes endpoint"""
    try:
        processed_count = process_resumes()
        return jsonify({
            "status": "success",
            "message": f"Processed {processed_count} new resumes",
            "knowledge_file": app.config['KNOWLEDGE_FILE']
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }), 500

@app.route('/')
def home():
    return "Resume Processor Running! Send POST requests to /Filter"

if __name__ == '__main__':
    app.run(debug=True)

# RAG Pipeline - Retrieval-Augmented Generation System

A complete RAG (Retrieval-Augmented Generation) pipeline in Python that processes PDFs and audio files, creates embeddings, stores them in a vector database, and enables intelligent question-answering using LLMs.

## Features

✨ **Complete Pipeline Components:**
- 📄 PDF text extraction
- 🎵 Audio transcription using OpenAI Whisper
- 🎬 MP4 video audio extraction and transcription
- ✂️ Smart text chunking with overlap
- 🔢 Vector embeddings using Sentence Transformers
- 💾 ChromaDB vector database for efficient similarity search
- 🤖 LLM integration (Google Gemini / OpenAI GPT / Anthropic Claude) for answer generation
- 🔍 Semantic search and retrieval

## Project Structure

```
rag_pipeline/
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment configuration
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── data/                        # Place your PDF and audio files here
│   ├── *.pdf
│   └── *.mp3, *.wav, etc.
└── modules/
    ├── __init__.py
    ├── pdf_loader.py            # PDF text extraction
    ├── audio_transcriber.py     # Audio to text transcription
    ├── text_chunker.py          # Text chunking logic
    ├── vector_database.py       # Vector DB and embeddings
    └── rag_system.py            # RAG query and generation
```

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- **ffmpeg** (required for audio/video processing)
- (Optional) GPU with CUDA for faster audio transcription

**Install ffmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 2. Clone or Navigate to Project

```bash
git clone https://github.com/novotnyradekcz/rag-pipeline
cd rag-pipeline
```

### 3. Create Virtual Environment

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
# venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing Whisper and PyTorch may take some time. If you have a GPU, install the CUDA version of PyTorch for faster transcription:

```bash
# For CUDA 11.8 (check your CUDA version):
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 5. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

**You need at least one LLM API key** for answer generation. The pipeline uses **Google Gemini by default** (you can also use OpenAI or Anthropic).

## Usage

### Prepare Your Data

Place your PDF and audio files in the `data/` directory:

```bash
# Example:
data/
├── research_paper.pdf
├── lecture_recording.mp3
├── video_lecture.mp4
└── notes.pdf
```

### Basic Usage

#### 1. Process All Files in Data Directory

```bash
python main.py
```

This will:
- Load all PDFs and audio files from `data/`
- Extract/transcribe text
- Create chunks
- Generate embeddings
- Store in vector database

#### 2. Ask a Question

```bash
python main.py --query "What is the main topic discussed?"
```

#### 3. Interactive Q&A Mode

```bash
python main.py --interactive
```

Then type your questions interactively. Type `quit` to exit.

### Advanced Usage

#### Process Specific Files

```bash
# Process only a specific PDF
python main.py --pdf data/research_paper.pdf --query "What are the key findings?"

# Process only a specific audio file
python main.py --audio data/lecture.mp3 --query "What topics were covered?"
```

#### Use Different LLM Provider

```bash
# Default is Google Gemini
python main.py --interactive

# Use OpenAI GPT instead
python main.py --llm openai --interactive

# Use Anthropic Claude
python main.py --llm anthropic --interactive
```

#### Reset Vector Database

```bash
# Clear the database and reprocess everything
python main.py --reset
```

#### Specify Custom Data Directory

```bash
python main.py --data-dir /path/to/your/files --interactive
```

## How It Works

### 1. Data Loading and Processing

**PDFs:** The pipeline uses PyPDF2 to extract text from PDF documents page by page.

**Audio:** OpenAI's Whisper model transcribes audio recordings to text. The default model is `base` (good balance of speed and accuracy).

### 2. Text Chunking

Text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`:
- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Splits on paragraphs, sentences, then words to maintain semantic meaning

### 3. Embedding Generation

Uses Sentence Transformers (`all-MiniLM-L6-v2` by default) to convert text chunks into 384-dimensional vectors. This model provides a good balance of speed and quality.

### 4. Vector Database Storage

ChromaDB stores the embeddings with metadata. It uses cosine similarity for efficient semantic search.

### 5. Retrieval and Generation

When you ask a question:
1. Query is converted to a vector
2. Top-k most similar chunks are retrieved from the database
3. Retrieved context + your question are sent to the LLM
4. LLM generates a contextual answer

## Configuration Options

### Customize in Code

Edit `main.py` or create your own script:

```python
from modules import RAGPipeline

pipeline = RAGPipeline(
    collection_name="my_collection",     # Database collection name
    chunk_size=800,                      # Smaller chunks
    chunk_overlap=150,                   # Less overlap
    embedding_model="all-mpnet-base-v2", # Better embeddings
    whisper_model="small",               # Better transcription
    llm_provider="gemini",               # Use Gemini
    llm_model="gemini-2.5-flash"         # Specific model
)

# Process files
chunks = pipeline.process_directory("./data")
pipeline.build_knowledge_base(chunks)

# Query
result = pipeline.query("Your question here?", n_results=3)
print(result['answer'])
```

## Module Documentation

### PDFLoader
```python
from modules.pdf_loader import PDFLoader

loader = PDFLoader("document.pdf")
text = loader.extract_text()
```

### AudioTranscriber
```python
from modules.audio_transcriber import AudioTranscriber

transcriber = AudioTranscriber(model_size="base")
result = transcriber.transcribe_audio("recording.mp3")
print(result['text'])
```

### TextChunker
```python
from modules.text_chunker import TextChunker

chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_text(text, metadata={'source': 'document.pdf'})
```

### VectorDatabase
```python
from modules.vector_database import VectorDatabase

db = VectorDatabase(collection_name="my_docs")
db.add_documents(chunks)
results = db.query("search query", n_results=5)
```

### RAGSystem
```python
from modules.rag_system import RAGSystem

rag = RAGSystem(vector_db=db, llm_provider="gemini")
result = rag.answer_question("What is discussed?")
print(result['answer'])
```

## Troubleshooting

### Whisper Model Download

First run will download the Whisper model (~150MB for base model). Ensure you have a stable internet connection.

### Memory Issues

If transcribing large audio files:
- Use a smaller Whisper model: `whisper_model="tiny"`
- Process audio in smaller segments
- Close other applications

### API Rate Limits

If you hit OpenAI/Anthropic rate limits:
- Add delays between requests
- Use a lower tier model
- Upgrade your API plan

### ChromaDB Persistence

The vector database is stored in `chroma_db/` directory. To start fresh:
```bash
rm -rf chroma_db/
python main.py --reset
```

## Performance Tips

1. **Whisper Model Selection:**
   - `tiny`: Fastest, least accurate
   - `base`: Good default (recommended)
   - `small`: Better accuracy, slower
   - `medium`/`large`: Best accuracy, much slower

2. **Embedding Model Selection:**
   - `all-MiniLM-L6-v2`: Fast, good quality (default)
   - `all-mpnet-base-v2`: Better quality, slower
   - `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A

3. **Chunk Size:**
   - Smaller (500-800): More precise retrieval, more chunks
   - Larger (1000-1500): More context per chunk, fewer chunks

4. **Number of Retrieved Chunks:**
   - Start with 3-5 chunks
   - Increase if answers lack context
   - Decrease if answers become unfocused

## Example Workflow

```bash
# 1. Setup
cd rag_pipeline
source venv/bin/activate

# 2. Add your files to data/
cp ~/my_document.pdf data/
cp ~/lecture.mp3 data/

# 3. Process and build knowledge base
python main.py

# 4. Ask questions
python main.py --query "Summarize the key points"

# 5. Interactive mode
python main.py --interactive
```

## Dependencies

Key libraries used:
- **PyPDF2**: PDF text extraction
- **openai-whisper**: Audio transcription
- **moviepy**: MP4 audio extraction
- **sentence-transformers**: Text embeddings
- **chromadb**: Vector database
- **langchain**: Text processing utilities
- **openai**: OpenAI GPT API
- **anthropic**: Anthropic Claude API
- **google-generativeai**: Google Gemini API

## License

This project is for educational purposes (Ciklum AI Academy HW4).

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your API keys are correct
3. Ensure all dependencies are installed
4. Check that your data files are in the correct format

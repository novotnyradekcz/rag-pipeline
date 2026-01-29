"""
Main RAG Pipeline
Complete pipeline that orchestrates all components.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
from datetime import datetime
import json

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.pdf_loader import PDFLoader
from modules.audio_transcriber import AudioTranscriber
from modules.text_chunker import TextChunker
from modules.vector_database import VectorDatabase
from modules.rag_system import RAGSystem


class RAGPipeline:
    """Complete RAG pipeline orchestrator."""
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        whisper_model: str = "base",
        llm_provider: str = "openai",
        llm_model: Optional[str] = None
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            collection_name: Name for the vector database collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence transformer model name
            whisper_model: Whisper model size for audio transcription
            llm_provider: 'openai' or 'anthropic'
            llm_model: Specific LLM model name
        """
        print("="*80)
        print("Initializing RAG Pipeline")
        print("="*80)
        
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print("✓ Text Chunker initialized")
        
        self.vector_db = VectorDatabase(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        print("✓ Vector Database initialized")
        
        self.audio_transcriber = None  # Lazy load
        self.whisper_model = whisper_model
        
        self.rag_system = RAGSystem(
            vector_db=self.vector_db,
            llm_provider=llm_provider,
            model_name=llm_model
        )
        print("✓ RAG System initialized")
        print("="*80)
    
    def load_pdf(self, pdf_path: str) -> List[dict]:
        """
        Load and process a PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of document chunks
        """
        print(f"\n📄 Loading PDF: {pdf_path}")
        loader = PDFLoader(pdf_path)
        text = loader.extract_text()
        
        print("✓ PDF loaded successfully")
        
        # Chunk the text
        chunks = self.text_chunker.chunk_text(
            text,
            metadata={'source': Path(pdf_path).name, 'type': 'pdf'}
        )
        
        stats = self.text_chunker.get_chunk_stats(chunks)
        print(f"✓ Created {stats['total_chunks']} chunks")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} characters")
        
        return chunks
    
    def transcribe_audio(self, audio_path: str) -> List[dict]:
        """
        Transcribe and process an audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of document chunks
        """
        print(f"\n🎵 Transcribing audio: {audio_path}")
        
        # Lazy load transcriber (it's heavy)
        if self.audio_transcriber is None:
            self.audio_transcriber = AudioTranscriber(model_size=self.whisper_model)
        
        result = self.audio_transcriber.transcribe_audio(audio_path)
        text = result['text']
        
        print("✓ Audio transcribed successfully")
        
        # Chunk the text
        chunks = self.text_chunker.chunk_text(
            text,
            metadata={'source': Path(audio_path).name, 'type': 'audio'}
        )
        
        stats = self.text_chunker.get_chunk_stats(chunks)
        print(f"✓ Created {stats['total_chunks']} chunks")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} characters")
        
        return chunks
    
    def process_directory(self, directory: str) -> List[dict]:
        """
        Process all PDF and audio files in a directory.
        
        Args:
            directory: Path to directory
        
        Returns:
            Combined list of all chunks
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_chunks = []
        
        # Process PDFs
        pdf_files = list(dir_path.glob("*.pdf"))
        for pdf_file in pdf_files:
            chunks = self.load_pdf(str(pdf_file))
            all_chunks.extend(chunks)
        
        # Process audio files (including MP4 video files)
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(dir_path.glob(f"*{ext}"))
        
        for audio_file in audio_files:
            chunks = self.transcribe_audio(str(audio_file))
            all_chunks.extend(chunks)
        
        print(f"\n✓ Processed {len(pdf_files)} PDFs and {len(audio_files)} audio files")
        print(f"✓ Total chunks: {len(all_chunks)}")
        
        return all_chunks
    
    def build_knowledge_base(self, chunks: List[dict]):
        """
        Add chunks to the vector database.
        
        Args:
            chunks: List of document chunks
        """
        print(f"\n🔧 Building knowledge base...")
        self.vector_db.add_documents(chunks)
        
        stats = self.vector_db.get_collection_stats()
        print(f"✓ Knowledge base ready!")
        print(f"  Total documents: {stats['total_documents']}")
    
    def query(self, question: str, n_results: int = 5) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            n_results: Number of context chunks to retrieve
        
        Returns:
            Dictionary with answer and context
        """
        return self.rag_system.answer_question(
            question,
            n_results=n_results,
            return_context=True
        )
    
    def interactive_mode(self):
        """Start interactive Q&A mode and save session transcript."""
        session_history = self.rag_system.interactive_qa()
        
        # Save session transcript if there were any queries
        if session_history:
            self._save_session_log(session_history)
    
    def _save_query_log(self, result: dict):
        """Save a single query and answer to a log file."""
        timestamp = datetime.now()
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        filename = f"query_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = log_dir / filename
        
        # Format log content
        log_content = []
        log_content.append("="*80)
        log_content.append(f"RAG Query Log")
        log_content.append(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append(f"LLM Provider: {self.rag_system.llm_provider}")
        log_content.append(f"Model: {self.rag_system.model_name}")
        log_content.append("="*80)
        log_content.append(f"\nQuery:\n{result['query']}\n")
        log_content.append(f"\nAnswer:\n{result['answer']}\n")
        
        if 'context' in result:
            log_content.append("\nSources:")
            for i, ctx in enumerate(result['context'], 1):
                source = ctx['metadata'].get('source', 'unknown')
                score = ctx['score']
                log_content.append(f"  {i}. {source} (relevance: {score:.3f})")
        
        log_content.append("\n" + "="*80)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))
        
        print(f"\n✓ Query log saved to: {filepath.name}")
    
    def _save_session_log(self, session_history: List[dict]):
        """Save interactive session transcript to a log file."""
        timestamp = datetime.now()
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = log_dir / filename
        
        # Format log content
        log_content = []
        log_content.append("="*80)
        log_content.append(f"RAG Interactive Session Transcript")
        log_content.append(f"Session Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append(f"LLM Provider: {self.rag_system.llm_provider}")
        log_content.append(f"Model: {self.rag_system.model_name}")
        log_content.append(f"Total Queries: {len(session_history)}")
        log_content.append("="*80)
        
        # Add each Q&A exchange
        for i, exchange in enumerate(session_history, 1):
            log_content.append(f"\n{'='*80}")
            log_content.append(f"Query #{i}")
            log_content.append(f"{'='*80}")
            log_content.append(f"\nQuestion:\n{exchange['query']}\n")
            log_content.append(f"\nAnswer:\n{exchange['answer']}\n")
            
            if 'sources' in exchange:
                log_content.append("\nSources:")
                for j, src in enumerate(exchange['sources'], 1):
                    log_content.append(f"  {j}. {src['source']} (relevance: {src['score']:.3f})")
        
        log_content.append("\n" + "="*80)
        log_content.append(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append("="*80)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))
        
        print(f"\n✓ Session transcript saved to: {filepath.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing PDF and audio files'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        help='Specific PDF file to process'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Specific audio file to process'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Question to ask'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive Q&A mode'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset the vector database'
    )
    parser.add_argument(        '--reindex',
        action='store_true',
        help='Force reindexing of all files (even if already processed)'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default='gemini',
        choices=['openai', 'anthropic', 'gemini'],
        help='LLM provider to use (default: gemini)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(llm_provider=args.llm)
    
    # Reset database if requested
    if args.reset:
        print("\n⚠️  Resetting vector database...")
        pipeline.vector_db.reset_collection()
        print("✓ Database reset complete")
    
    # Check if database already has documents
    stats = pipeline.vector_db.get_collection_stats()
    has_documents = stats['total_documents'] > 0
    
    # Process data only if:
    # 1. Database is empty, OR
    # 2. User explicitly requests reindexing with --reindex, OR
    # 3. User specifies specific files with --pdf or --audio
    should_process = (
        not has_documents or 
        args.reindex or 
        args.pdf or 
        args.audio
    )
    
    if should_process:
        # Process data
        all_chunks = []
        
        if args.pdf:
            chunks = pipeline.load_pdf(args.pdf)
            all_chunks.extend(chunks)
        
        if args.audio:
            chunks = pipeline.transcribe_audio(args.audio)
            all_chunks.extend(chunks)
        
        if not args.pdf and not args.audio and os.path.exists(args.data_dir):
            chunks = pipeline.process_directory(args.data_dir)
            all_chunks.extend(chunks)
        
        # Build knowledge base if we have chunks
        if all_chunks:
            pipeline.build_knowledge_base(all_chunks)
    else:
        print(f"\n✓ Using existing database with {stats['total_documents']} documents")
        print("  Use --reindex to reprocess all files")
        print("  Use --reset to clear the database")
    
    # Query or interactive mode
    if args.query:
        result = pipeline.query(args.query)
        print(f"\n{'='*80}")
        print(f"Question: {result['query']}")
        print(f"{'='*80}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\n{'='*80}")
        print("Sources:")
        for i, ctx in enumerate(result['context'], 1):
            print(f"  {i}. {ctx['metadata']['source']} (score: {ctx['score']:.3f})")
        
        # Save query log
        pipeline._save_query_log(result)
    
    elif args.interactive:
        pipeline.interactive_mode()
    
    else:
        print("\n✓ Pipeline ready!")
        print("Use --query 'your question' to ask a question")
        print("Use --interactive for interactive mode")


if __name__ == "__main__":
    main()

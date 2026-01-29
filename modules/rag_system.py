"""
RAG (Retrieval-Augmented Generation) Module
Retrieves relevant context and generates answers using LLM.
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGSystem:
    """Handles retrieval and generation for question answering."""
    
    def __init__(
        self,
        vector_db,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize RAG system.
        
        Args:
            vector_db: Instance of VectorDatabase
            llm_provider: 'openai', 'anthropic', or 'gemini'
            model_name: Model name (default: gpt-3.5-turbo for OpenAI, claude-3-sonnet for Anthropic, gemini-2.5-flash for Gemini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.vector_db = vector_db
        self.llm_provider = llm_provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM client
        if self.llm_provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name or "gpt-3.5-turbo"
        elif self.llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = model_name or "claude-3-sonnet-20240229"
        elif self.llm_provider == "gemini":
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            # Use the latest stable Gemini model (gemini-2.5-flash is faster, gemini-2.5-pro is more capable)
            self.model_name = model_name or "gemini-2.5-flash"
            self.client = genai.GenerativeModel(self.model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Choose 'openai', 'anthropic', or 'gemini'")
        
        print(f"Initialized RAG system with {self.llm_provider} ({self.model_name})")
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant context from vector database.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
        
        Returns:
            List of retrieved chunks with metadata and scores
        """
        results = self.vector_db.query(query, n_results=n_results)
        
        retrieved_chunks = []
        for i in range(len(results['documents'])):
            retrieved_chunks.append({
                'text': results['documents'][i],
                'score': 1 - results['distances'][i],  # Convert distance to similarity
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            })
        
        return retrieved_chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'unknown')
            context_parts.append(f"[Context {i} from {source}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
            system_prompt: Custom system prompt (optional)
        
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use the context to provide accurate and detailed answers. "
                "If the context doesn't contain enough information to answer the question, "
                "say so honestly and provide what information you can."
            )
        
        user_message = f"""Context:
{context}

Question: {query}

Please answer the question based on the context provided above."""
        
        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                return response.content[0].text
            
            elif self.llm_provider == "gemini":
                # Gemini combines system prompt and user message
                full_prompt = f"{system_prompt}\n\n{user_message}"
                
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                )
                return response.text
        
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def answer_question(
        self,
        query: str,
        n_results: int = 5,
        return_context: bool = False
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            return_context: Whether to return retrieved context
        
        Returns:
            Dictionary with answer and optionally context
        """
        print(f"\nProcessing query: {query}")
        
        # Retrieve relevant context
        print(f"Retrieving top {n_results} relevant chunks...")
        chunks = self.retrieve_context(query, n_results)
        
        # Format context
        context = self.format_context(chunks)
        
        # Generate answer
        print("Generating answer...")
        answer = self.generate_answer(query, context)
        
        result = {
            'query': query,
            'answer': answer
        }
        
        if return_context:
            result['context'] = chunks
        
        return result
    
    def interactive_qa(self):
        """
        Interactive question-answering loop with session tracking.
        Returns the session history for logging.
        """
        print("\n=== RAG Interactive Q&A ===")
        print("Type your question or 'quit' to exit.\n")
        
        session_history = []
        
        while True:
            query = input("Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            result = self.answer_question(query, return_context=True)
            
            print(f"\n{'-'*80}")
            print(f"Answer: {result['answer']}")
            print(f"{'-'*80}")
            
            # Show sources
            print("\nSources:")
            sources = []
            for i, chunk in enumerate(result['context'], 1):
                score = chunk['score']
                source = chunk['metadata'].get('source', 'unknown')
                print(f"  {i}. {source} (relevance: {score:.2f})")
                sources.append({'source': source, 'score': score})
            print()
            
            # Track session history
            session_history.append({
                'query': query,
                'answer': result['answer'],
                'sources': sources
            })
        
        return session_history


if __name__ == "__main__":
    from vector_database import VectorDatabase
    
    # Initialize components
    db = VectorDatabase()
    
    # Create RAG system (defaults to OpenAI)
    rag = RAGSystem(vector_db=db, llm_provider="openai")
    
    # Example query
    result = rag.answer_question(
        "What is machine learning?",
        n_results=3,
        return_context=True
    )
    
    print(f"\nAnswer: {result['answer']}")

"""
Autonomous Deep Research Agent
Multi-agent system with planner-executor architecture using LangGraph
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json
import re
from dataclasses import dataclass, field
from collections import defaultdict

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Vector storage and embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Web scraping
from bs4 import BeautifulSoup
import trafilatura


@dataclass
class Citation:
    """Tracks source citations"""
    source_id: str
    url: str
    title: str
    snippet: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResearchSubTask:
    """Represents a decomposed research subtask"""
    id: str
    query: str
    priority: int
    status: str = "pending"  # pending, in_progress, completed, failed
    results: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)


class AgentState(TypedDict):
    """State shared across all agents"""
    original_query: str
    subtasks: List[ResearchSubTask]
    completed_subtasks: List[ResearchSubTask]
    research_plan: str
    raw_sources: List[Dict[str, Any]]
    indexed_documents: List[Dict[str, Any]]
    citations: List[Citation]
    final_report: str
    current_step: str
    error: Optional[str]


class DeepResearchAgent:
    """Main orchestrator for the deep research system"""
    
    def __init__(self, gemini_api_key: str, max_sources: int = 20):
        self.gemini_api_key = gemini_api_key
        self.max_sources = max_sources
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.7
        )
        
        # Initialize embedding model for vector search
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384
        self.index = None
        self.document_store = []
        
        # Build the agent graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Constructs the LangGraph multi-agent workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("executor", self.executor_agent)
        workflow.add_node("indexer", self.indexer_agent)
        workflow.add_node("synthesizer", self.synthesizer_agent)
        
        # Define the workflow edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "indexer")
        workflow.add_edge("indexer", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def planner_agent(self, state: AgentState) -> AgentState:
        """
        Decomposes complex queries into parallel subtasks
        """
        print("🧠 PLANNER: Decomposing query into subtasks...")
        
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research planning expert. Given a complex query, decompose it into 3-7 specific, 
            focused subtasks that can be researched in parallel. Each subtask should:
            1. Be specific and actionable
            2. Cover a distinct aspect of the main query
            3. Be researchable through web sources
            
            Return a JSON array of subtasks with this format:
            [
                {{"id": "task_1", "query": "specific research question", "priority": 1}},
                {{"id": "task_2", "query": "another specific question", "priority": 2}}
            ]
            
            Only return the JSON array, nothing else."""),
            ("human", "Main query: {query}")
        ])
        
        messages = planning_prompt.format_messages(query=state["original_query"])
        response = await self.llm.ainvoke(messages)
        
        # Parse the subtasks
        try:
            subtasks_data = json.loads(response.content)
            subtasks = [
                ResearchSubTask(
                    id=task["id"],
                    query=task["query"],
                    priority=task["priority"]
                ) for task in subtasks_data
            ]
            
            state["subtasks"] = subtasks
            state["research_plan"] = f"Decomposed into {len(subtasks)} subtasks"
            state["current_step"] = "planning_complete"
            
            print(f"✅ Created {len(subtasks)} subtasks")
            for task in subtasks:
                print(f"   - {task.id}: {task.query}")
                
        except json.JSONDecodeError as e:
            state["error"] = f"Failed to parse subtasks: {e}"
            print(f"❌ Error: {state['error']}")
            
        return state
    
    async def executor_agent(self, state: AgentState) -> AgentState:
        """
        Executes research subtasks in parallel using async I/O
        Implements high-concurrency ingestion pipeline
        """
        print("\n🔍 EXECUTOR: Running parallel research tasks...")
        
        subtasks = state["subtasks"]
        
        # Execute all subtasks concurrently
        results = await asyncio.gather(
            *[self._execute_single_task(task) for task in subtasks],
            return_exceptions=True
        )
        
        # Process results
        all_sources = []
        all_citations = []
        completed_tasks = []
        
        for task, result in zip(subtasks, results):
            if isinstance(result, Exception):
                print(f"❌ Task {task.id} failed: {result}")
                task.status = "failed"
            else:
                task.status = "completed"
                task.results = result["sources"]
                task.citations = result["citations"]
                all_sources.extend(result["sources"])
                all_citations.extend(result["citations"])
                completed_tasks.append(task)
                print(f"✅ Task {task.id} completed: {len(result['sources'])} sources")
        
        state["completed_subtasks"] = completed_tasks
        state["raw_sources"] = all_sources[:self.max_sources]  # Limit to max sources
        state["citations"] = all_citations
        state["current_step"] = "execution_complete"
        
        print(f"\n📊 Total sources collected: {len(all_sources)}")
        
        return state
    
    async def _execute_single_task(self, task: ResearchSubTask) -> Dict[str, Any]:
        """
        Executes a single research task with web search and scraping
        """
        task.status = "in_progress"
        
        # Simulate web search API calls (replace with actual API)
        sources = await self._search_web(task.query)
        
        # Scrape content from sources in parallel
        scraped_content = await asyncio.gather(
            *[self._scrape_url(source["url"]) for source in sources[:5]],
            return_exceptions=True
        )
        
        # Process and create citations
        citations = []
        valid_sources = []
        
        for i, (source, content) in enumerate(zip(sources[:5], scraped_content)):
            if not isinstance(content, Exception) and content:
                citation = Citation(
                    source_id=f"{task.id}_src_{i}",
                    url=source["url"],
                    title=source.get("title", "Untitled"),
                    snippet=content[:200]
                )
                citations.append(citation)
                
                valid_sources.append({
                    "task_id": task.id,
                    "url": source["url"],
                    "title": source.get("title", ""),
                    "content": content,
                    "citation": citation
                })
        
        return {
            "sources": valid_sources,
            "citations": citations
        }
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Simulates web search API (replace with actual Google Custom Search, Bing, etc.)
        
        NOTE: This is a MOCK implementation for testing. 
        For production, integrate with real search APIs:
        - Google Custom Search API
        - Bing Search API  
        - SerpAPI
        - DuckDuckGo API
        
        See enhanced_agent.py for real implementations.
        """
        
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Return mock results with realistic content based on the query
        import hashlib
        query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
        
        mock_results = []
        for i in range(5):
            seed = query_hash + i
            mock_results.append({
                "url": f"https://example-research-{seed % 1000}.com/article-{i}",
                "title": f"Research Article {i+1}: {query[:50]}",
                "snippet": f"This article discusses {query}. Key findings include important insights about the topic, with detailed analysis and expert opinions on the subject matter."
            })
        
        return mock_results
    
    async def _scrape_url(self, url: str) -> Optional[str]:
        """
        Scrapes content from a URL using trafilatura
        
        NOTE: This mock version returns sample content for testing.
        For production, this will fetch real web pages.
        """
        try:
            # For testing without real URLs, return mock content
            if "example" in url or "mock" in url:
                await asyncio.sleep(0.05)  # Simulate network delay
                
                # Generate realistic mock content based on URL
                import hashlib
                url_hash = int(hashlib.md5(url.encode()).hexdigest(), 16) % 10
                
                mock_contents = [
                    "Quantum computing represents a paradigm shift in computational technology. Unlike classical computers that use bits (0 or 1), quantum computers use qubits which can exist in superposition. This allows quantum computers to process multiple states simultaneously, potentially solving certain complex problems exponentially faster than classical computers.",
                    
                    "The pharmaceutical industry stands to benefit significantly from quantum computing. Drug discovery, which currently takes years and costs billions, could be accelerated through quantum simulations of molecular interactions. Quantum algorithms can model protein folding and chemical reactions with unprecedented accuracy.",
                    
                    "Major technical challenges in quantum computing include maintaining qubit coherence and developing effective error correction methods. Qubits are extremely sensitive to environmental interference, a phenomenon called decoherence. Current quantum computers require near-absolute-zero temperatures and sophisticated isolation to maintain quantum states.",
                    
                    "The economic barriers to quantum computing are substantial. Building and maintaining quantum computers requires specialized facilities, cryogenic cooling systems, and expert personnel. Current costs can exceed hundreds of millions of dollars for a single quantum computing system. Scalability remains a significant challenge.",
                    
                    "The quantum computing workforce faces a severe talent shortage. The field requires expertise in quantum physics, computer science, and engineering. Universities are rushing to develop quantum computing curricula, but the gap between demand and supply of qualified professionals continues to widen.",
                    
                    "Quantum computers pose a serious threat to current encryption standards. They can potentially break RSA and elliptic curve cryptography that secure most internet communications. This has spurred development of post-quantum cryptography - new encryption methods resistant to quantum attacks. NIST is currently standardizing post-quantum cryptographic algorithms.",
                    
                    "Machine learning and artificial intelligence could be revolutionized by quantum computing. Quantum machine learning algorithms promise faster training of neural networks and better optimization. Companies like IBM, Google, and Microsoft are developing quantum-enhanced AI tools.",
                    
                    "Financial services are exploring quantum computing for portfolio optimization, risk analysis, and fraud detection. Quantum algorithms could analyze market data and identify patterns that classical computers miss. However, practical implementation is still years away.",
                    
                    "Quantum computing applications in logistics include route optimization, supply chain management, and scheduling problems. These optimization challenges grow exponentially complex with classical computers but could be efficiently solved by quantum algorithms.",
                    
                    "The current state of quantum computing is often described as the 'NISQ era' - Noisy Intermediate-Scale Quantum. We have quantum computers with 50-1000 qubits, but they're error-prone and limited. Achieving fault-tolerant quantum computing with millions of qubits remains a distant goal requiring breakthrough innovations."
                ]
                
                return mock_contents[url_hash]
            
            # For real URLs (when integrated with actual search)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Extract main content using trafilatura
                        content = trafilatura.extract(html)
                        return content or ""
        except Exception as e:
            print(f"⚠️  Failed to scrape {url}: {e}")
            return None
    
    async def indexer_agent(self, state: AgentState) -> AgentState:
        """
        Indexes documents in FAISS vector store for efficient retrieval
        """
        print("\n📚 INDEXER: Building vector index...")
        
        documents = state["raw_sources"]
        
        if not documents:
            print("⚠️  No documents to index - using fallback mode")
            state["indexed_documents"] = []
            state["current_step"] = "indexing_complete"
            state["error"] = "No sources collected - search functionality may not be working"
            return state
        
        # Extract text content and create embeddings
        texts = []
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            texts.append(text)
        
        # Generate embeddings in batches for efficiency
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents with metadata
        self.document_store = documents
        
        state["indexed_documents"] = documents
        state["current_step"] = "indexing_complete"
        
        print(f"✅ Indexed {len(documents)} documents")
        
        return state
    
    async def synthesizer_agent(self, state: AgentState) -> AgentState:
        """
        Synthesizes final report with citation-grounded outputs
        """
        print("\n✍️  SYNTHESIZER: Generating final report...")
        
        # Check if we have an index and documents
        if self.index is None or not self.document_store:
            print("⚠️  No sources available - generating report based on LLM knowledge")
            
            # Generate a report without sources
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research expert. The user asked for research on a topic, 
                but no web sources were available. Please provide a comprehensive overview based on 
                your training knowledge. Be clear that this is based on general knowledge, not 
                real-time research."""),
                ("human", """Query: {query}
                
Please provide a comprehensive response based on your knowledge. Include:
1. An overview of the topic
2. Key points and considerations
3. A note that this is based on general knowledge, not live research

Be thorough but acknowledge the limitation of not having current sources.""")
            ])
            
            messages = fallback_prompt.format_messages(query=state["original_query"])
            response = await self.llm.ainvoke(messages)
            
            final_report = f"""# Research Report

**Query:** {state["original_query"]}

**Note:** This report is based on general knowledge as no web sources were available for real-time research.

---

{response.content}

---

**Recommendation:** To get real-time research with citations, please ensure:
1. You have internet connectivity
2. The search functionality is properly configured
3. Consider integrating a real search API (Google Custom Search, SerpAPI, etc.)
"""
            
            state["final_report"] = final_report
            state["current_step"] = "complete"
            
            print("✅ Fallback report generated (no sources available)")
            return state
        
        # Normal path with sources
        # Retrieve relevant passages using vector similarity
        query_embedding = self.embedder.encode([state["original_query"]])
        D, I = self.index.search(query_embedding.astype('float32'), k=min(10, len(self.document_store)))
        
        relevant_docs = [self.document_store[i] for i in I[0] if i < len(self.document_store)]
        
        # Prepare context with citations
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            citation_marker = f"[{i+1}]"
            context_parts.append(
                f"{citation_marker} {doc.get('title', '')}\n{doc.get('content', '')[:500]}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate synthesis
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research synthesis expert. Based on the provided sources, create a comprehensive 
            report that answers the original query. 
            
            IMPORTANT: 
            - Use inline citations [1], [2], etc. to reference sources
            - Every claim must be citation-grounded
            - Structure the report with clear sections
            - Be comprehensive but concise
            - Synthesize information across sources"""),
            ("human", """Original Query: {query}
            
Sources:
{context}

Generate a comprehensive research report:""")
        ])
        
        messages = synthesis_prompt.format_messages(
            query=state["original_query"],
            context=context
        )
        
        response = await self.llm.ainvoke(messages)
        
        # Create bibliography
        bibliography = "\n\nReferences:\n"
        for i, doc in enumerate(relevant_docs):
            citation = doc.get("citation")
            if citation:
                bibliography += f"[{i+1}] {citation.title} - {citation.url}\n"
        
        final_report = response.content + bibliography
        
        state["final_report"] = final_report
        state["current_step"] = "complete"
        
        print("✅ Report generation complete")
        
        return state
    
    async def research(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for research agent
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Deep Research on: {query}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        # Initialize state
        initial_state: AgentState = {
            "original_query": query,
            "subtasks": [],
            "completed_subtasks": [],
            "research_plan": "",
            "raw_sources": [],
            "indexed_documents": [],
            "citations": [],
            "final_report": "",
            "current_step": "initialized",
            "error": None
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"✅ Research Complete in {duration:.2f} seconds")
        print(f"{'='*80}\n")
        
        return {
            "query": query,
            "report": final_state["final_report"],
            "subtasks": len(final_state["completed_subtasks"]),
            "sources": len(final_state["raw_sources"]),
            "citations": len(final_state["citations"]),
            "duration_seconds": duration,
            "state": final_state
        }


async def main():
    """Demo usage"""
    import os
    
    # Get API key from environment or prompt
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ")
    
    # Initialize agent
    agent = DeepResearchAgent(gemini_api_key=api_key, max_sources=20)
    
    # Example research query
    query = "What are the latest developments in quantum computing and their potential impact on cryptography?"
    
    # Run research
    result = await agent.research(query)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESEARCH REPORT")
    print("="*80 + "\n")
    print(result["report"])
    print("\n" + "="*80)
    print(f"Statistics:")
    print(f"  - Subtasks: {result['subtasks']}")
    print(f"  - Sources: {result['sources']}")
    print(f"  - Citations: {result['citations']}")
    print(f"  - Duration: {result['duration_seconds']:.2f}s")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

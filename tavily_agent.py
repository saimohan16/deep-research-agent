"""
Tavily-Powered Deep Research Agent
Uses Tavily Search API for real-time web research
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from tavily import TavilyClient
from deep_research_agent import DeepResearchAgent, AgentState, ResearchSubTask, Citation


class TavilyResearchAgent(DeepResearchAgent):
    """
    Deep Research Agent powered by Tavily Search API
    
    Tavily is specifically designed for AI agents and provides:
    - Clean, structured search results
    - Automatic content extraction
    - Answer-focused responses
    - High relevance filtering
    """
    
    def __init__(
        self, 
        gemini_api_key: str,
        tavily_api_key: Optional[str] = None,
        max_sources: int = 20,
        search_depth: str = "advanced"  # "basic" or "advanced"
    ):
        super().__init__(gemini_api_key, max_sources)
        
        # Initialize Tavily client
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError(
                "Tavily API key required. Get one free at https://tavily.com\n"
                "Set TAVILY_API_KEY environment variable or pass tavily_api_key parameter"
            )
        
        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.search_depth = search_depth
        
        print(f"🔍 Tavily Search initialized (depth: {search_depth})")
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Tavily API
        """
        try:
            # Tavily search is synchronous, so run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(
                    query=query,
                    search_depth=self.search_depth,
                    max_results=10,
                    include_raw_content=True,  # Get full content
                    include_answer=False  # We'll synthesize our own
                )
            )
            
            results = []
            for item in response.get('results', []):
                results.append({
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'snippet': item.get('content', ''),  # Tavily provides clean content
                    'raw_content': item.get('raw_content', ''),  # Full page content
                    'score': item.get('score', 0)  # Relevance score
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️  Tavily search failed for '{query}': {e}")
            return []
    
    async def _scrape_url(self, url: str) -> Optional[str]:
        """
        With Tavily, we already have the content from the search results
        So this is much simpler - we just return the raw_content
        """
        # This is called during execution, but Tavily already gave us the content
        # So we'll just return a placeholder and use the content from search results
        return "Content provided by Tavily search"
    
    async def _execute_single_task(self, task: ResearchSubTask) -> Dict[str, Any]:
        """
        Override to use Tavily's pre-extracted content
        """
        task.status = "in_progress"
        
        # Search with Tavily
        sources = await self._search_web(task.query)
        
        # Create citations and process sources
        citations = []
        valid_sources = []
        
        for i, source in enumerate(sources[:5]):
            # Use Tavily's extracted content directly
            content = source.get('raw_content') or source.get('snippet', '')
            
            if content:
                citation = Citation(
                    source_id=f"{task.id}_src_{i}",
                    url=source['url'],
                    title=source.get('title', 'Untitled'),
                    snippet=source.get('snippet', '')[:200]
                )
                citations.append(citation)
                
                valid_sources.append({
                    "task_id": task.id,
                    "url": source['url'],
                    "title": source.get('title', ''),
                    "content": content,
                    "citation": citation,
                    "relevance_score": source.get('score', 0)
                })
        
        task.status = "completed"
        
        return {
            "sources": valid_sources,
            "citations": citations
        }


async def main():
    """Demo usage with Tavily"""
    
    # Get API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not gemini_key:
        gemini_key = input("Enter your Gemini API key: ")
    
    if not tavily_key:
        print("\n🔑 Tavily API Key needed")
        print("Get a FREE API key at: https://tavily.com")
        print("Free tier: 1000 searches/month")
        tavily_key = input("\nEnter your Tavily API key: ")
    
    # Initialize Tavily-powered agent
    print("\n" + "="*80)
    print("🚀 Tavily-Powered Deep Research Agent")
    print("="*80)
    
    agent = TavilyResearchAgent(
        gemini_api_key=gemini_key,
        tavily_api_key=tavily_key,
        max_sources=20,
        search_depth="advanced"  # Use "basic" for faster, "advanced" for deeper
    )
    
    # Example queries
    queries = [
        "What are the latest developments in quantum computing?",
        "How is AI being used in drug discovery?",
        "What are the main challenges in renewable energy storage?"
    ]
    
    print("\nAvailable demo queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    choice = input("\nSelect a query (1-3) or enter your own: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(queries):
        query = queries[int(choice) - 1]
    else:
        query = choice if choice else queries[0]
    
    # Run research
    print(f"\n{'='*80}")
    print(f"Researching: {query}")
    print(f"{'='*80}\n")
    
    result = await agent.research(query)
    
    # Display results
    print("\n" + "="*80)
    print("RESEARCH REPORT")
    print("="*80 + "\n")
    print(result["report"])
    print("\n" + "="*80)
    print(f"📊 Statistics:")
    print(f"  - Subtasks: {result['subtasks']}")
    print(f"  - Sources: {result['sources']}")
    print(f"  - Citations: {result['citations']}")
    print(f"  - Duration: {result['duration_seconds']:.2f}s")
    print("="*80)
    
    # Save results
    save = input("\nSave report to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(f"# Research Report\n\n")
            f.write(f"**Query:** {query}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
            f.write(result["report"])
        print(f"✅ Saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())

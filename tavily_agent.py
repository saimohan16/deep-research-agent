"""
Tavily-Powered Deep Research Agent
Uses Tavily Search API for real-time web research with automatic content extraction
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from tavily import TavilyClient
from deep_research_agent import DeepResearchAgent, AgentState, ResearchSubTask, Citation


class TavilyResearchAgent(DeepResearchAgent):
    """
    Deep Research Agent powered by Tavily Search API for production-ready web research.
    
    Tavily is specifically designed for AI agents and provides significant advantages
    over generic search APIs:
    
    Key Benefits:
        - Clean, structured search results optimized for LLM consumption
        - Automatic content extraction (no separate scraping needed)
        - Answer-focused responses with high relevance filtering
        - Built-in relevance scoring for result prioritization
        - Handles rate limiting and retries automatically
    
    This agent extends the base DeepResearchAgent, replacing the mock search
    implementation with real Tavily API calls while maintaining the same
    multi-agent workflow architecture.
    
    Architecture Integration:
        The planner and synthesizer remain unchanged from the base agent.
        Only the search and scraping components are overridden to use Tavily,
        demonstrating the pluggable design of the research agent system.
    
    Attributes:
        tavily_api_key: API key for Tavily search service
        tavily: Initialized TavilyClient instance
        search_depth: Either "basic" (faster) or "advanced" (more thorough)
    
    Example:
        agent = TavilyResearchAgent(
            gemini_api_key="your-gemini-key",
            tavily_api_key="your-tavily-key",
            search_depth="advanced"
        )
        result = await agent.research("Latest AI developments in healthcare")
    
    Note:
        Get a free API key at https://tavily.com (1000 searches/month free tier)
    """
    
    def __init__(
        self, 
        gemini_api_key: str,
        tavily_api_key: Optional[str] = None,
        max_sources: int = 20,
        search_depth: str = "advanced"  # "basic" or "advanced"
    ):
        """
        Initialize the Tavily-powered research agent.
        
        Args:
            gemini_api_key: Google Gemini API key for LLM operations
            tavily_api_key: Tavily API key (can also be set as TAVILY_API_KEY env var)
            max_sources: Maximum number of sources to collect per research task
            search_depth: Search depth - "basic" (faster) or "advanced" (more thorough)
            
        Raises:
            ValueError: If Tavily API key is not provided or found in environment
            
        Example:
            agent = TavilyResearchAgent(
                gemini_api_key="xxx",
                tavily_api_key="yyy",
                search_depth="advanced"
            )
        """
        super().__init__(gemini_api_key, max_sources)
        
        # Initialize Tavily client with API key from parameter or environment
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError(
                "Tavily API key required. Get one free at https://tavily.com\n"
                "Set TAVILY_API_KEY environment variable or pass tavily_api_key parameter"
            )
        
        # Create Tavily client for API interactions
        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.search_depth = search_depth
        
        print(f"🔍 Tavily Search initialized (depth: {search_depth})")
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search using Tavily API with automatic content extraction.
        
        Tavily provides several advantages over traditional search APIs:
        1. Automatic content extraction - no separate scraping needed
        2. Relevance scoring for each result
        3. Clean, LLM-friendly content format
        4. Built-in answer generation (optional, we disable for custom synthesis)
        
        The search is performed synchronously in a thread pool executor to avoid
        blocking the asyncio event loop, as the Tavily client is not natively async.
        
        Args:
            query: Search query string from a research subtask
            
        Returns:
            List of dicts containing search results with keys:
                - url: Source URL
                - title: Page title
                - snippet: Clean extracted content snippet
                - raw_content: Full page content (if include_raw_content=True)
                - score: Relevance score from Tavily (0-1)
                
        Example:
            results = await agent._search_web("quantum computing breakthroughs")
            for result in results:
                print(f"URL: {result['url']}, Score: {result['score']}")
        
        Note:
            This overrides the base class's mock implementation with real API calls.
        """
        try:
            # Tavily search is synchronous, so run in executor to not block event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(
                    query=query,
                    search_depth=self.search_depth,  # "basic" or "advanced"
                    max_results=10,  # Get top 10 results
                    include_raw_content=True,  # Get full page content for better context
                    include_answer=False  # We'll synthesize our own answer with Gemini
                )
            )
            
            # Transform Tavily response to our internal format
            results = []
            for item in response.get('results', []):
                results.append({
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'snippet': item.get('content', ''),  # Tavily provides clean content
                    'raw_content': item.get('raw_content', ''),  # Full page content
                    'score': item.get('score', 0)  # Relevance score (0-1)
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️  Tavily search failed for '{query}': {e}")
            return []  # Return empty list on failure to gracefully continue
    
    async def _scrape_url(self, url: str) -> Optional[str]:
        """
        Override scraping method since Tavily already provides content.
        
        With Tavily, we don't need separate web scraping because the API returns
        extracted content directly in the search results. This method exists only
        to satisfy the base class interface and is never actually called when
        using Tavily search results.
        
        Args:
            url: Web page URL (ignored in this implementation)
            
        Returns:
            Placeholder string indicating content is from Tavily
            
        Note:
            This method is overridden but not used. The actual content comes
            directly from the Tavily search results in _search_web().
        """
        # Tavily already gave us the content in the search results
        # So we just return a placeholder for interface compatibility
        return "Content provided by Tavily search"
    
    async def _execute_single_task(self, task: ResearchSubTask) -> Dict[str, Any]:
        """
        Execute a research subtask using Tavily's pre-extracted content.
        
        This override optimizes the execution flow by:
        1. Using Tavily's built-in content extraction (no separate scraping)
        2. Leveraging Tavily's relevance scores for result quality
        3. Processing results more efficiently without async scraping overhead
        
        Args:
            task: ResearchSubTask containing the specific query to research
            
        Returns:
            Dict containing:
                - sources: List of source documents with content and metadata
                - citations: List of Citation objects for proper attribution
                
        Example:
            Returns structure:
            {
                "sources": [
                    {
                        "task_id": "task_1",
                        "url": "https://...",
                        "title": "Article Title",
                        "content": "Full article content...",
                        "citation": Citation(...),
                        "relevance_score": 0.95
                    }
                ],
                "citations": [Citation(...), ...]
            }
        """
        task.status = "in_progress"
        
        # Search with Tavily - content is automatically extracted
        sources = await self._search_web(task.query)
        
        # Create citations and process sources using Tavily's pre-extracted content
        citations = []
        valid_sources = []
        
        for i, source in enumerate(sources[:5]):  # Limit to top 5 sources per task
            # Use Tavily's extracted content directly (no scraping needed)
            content = source.get('raw_content') or source.get('snippet', '')
            
            if content:  # Only include sources with actual content
                # Create citation for attribution
                citation = Citation(
                    source_id=f"{task.id}_src_{i}",
                    url=source['url'],
                    title=source.get('title', 'Untitled'),
                    snippet=source.get('snippet', '')[:200]  # First 200 chars as preview
                )
                citations.append(citation)
                
                # Store source with metadata including Tavily's relevance score
                valid_sources.append({
                    "task_id": task.id,
                    "url": source['url'],
                    "title": source.get('title', ''),
                    "content": content,
                    "citation": citation,
                    "relevance_score": source.get('score', 0)  # 0-1 relevance score
                })
        
        task.status = "completed"
        
        return {
            "sources": valid_sources,
            "citations": citations
        }


async def main():
    """
    Demonstration function for the Tavily-powered research agent.
    
    This function provides an interactive demo that:
    1. Validates and retrieves API keys (Gemini + Tavily)
    2. Offers pre-defined research queries or custom input
    3. Executes the research pipeline with real web search
    4. Displays the final report with statistics
    5. Optionally saves the report to a markdown file
    
    Environment Variables:
        GEMINI_API_KEY: Optional - can be set instead of interactive input
        TAVILY_API_KEY: Optional - can be set instead of interactive input
    
    Pre-defined Queries:
        1. Latest quantum computing developments
        2. AI applications in drug discovery
        3. Renewable energy storage challenges
    
    Output:
        - Displays research report in console
        - Optionally saves to timestamped markdown file
        - Shows execution statistics (sources, citations, duration)
    
    Example:
        python tavily_research_agent.py
        # Enter API keys when prompted or use environment variables
        # Select query 1, 2, 3, or enter custom question
        # Review report and optionally save to file
    """
    
    # Retrieve API keys from environment or prompt user
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not gemini_key:
        gemini_key = input("Enter your Gemini API key: ")
    
    if not tavily_key:
        print("\n🔑 Tavily API Key needed")
        print("Get a FREE API key at: https://tavily.com")
        print("Free tier: 1000 searches/month")
        tavily_key = input("\nEnter your Tavily API key: ")
    
    # Initialize Tavily-powered agent with real search capabilities
    print("\n" + "="*80)
    print("🚀 Tavily-Powered Deep Research Agent")
    print("="*80)
    
    agent = TavilyResearchAgent(
        gemini_api_key=gemini_key,
        tavily_api_key=tavily_key,
        max_sources=20,  # Limit to 20 sources total
        search_depth="advanced"  # "basic" for faster, "advanced" for deeper research
    )
    
    # Example queries showcasing different research domains
    queries = [
        "What are the latest developments in quantum computing?",
        "How is AI being used in drug discovery?",
        "What are the main challenges in renewable energy storage?"
    ]
    
    print("\nAvailable demo queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    # Get user query selection or custom input
    choice = input("\nSelect a query (1-3) or enter your own: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(queries):
        query = queries[int(choice) - 1]
    else:
        query = choice if choice else queries[0]  # Default to first query if empty
    
    # Execute the research pipeline
    print(f"\n{'='*80}")
    print(f"Researching: {query}")
    print(f"{'='*80}\n")
    
    result = await agent.research(query)
    
    # Display comprehensive results
    print("\n" + "="*80)
    print("RESEARCH REPORT")
    print("="*80 + "\n")
    print(result["report"])
    print("\n" + "="*80)
    print(f"📊 Statistics:")
    print(f"  - Subtasks: {result['subtasks']}")  # Number of parallel research tasks
    print(f"  - Sources: {result['sources']}")     # Total sources collected
    print(f"  - Citations: {result['citations']}") # Citations created
    print(f"  - Duration: {result['duration_seconds']:.2f}s")  # Total execution time
    print("="*80)
    
    # Optionally save report to file for later reference
    save = input("\nSave report to file? (y/n): ").strip().lower()
    if save == 'y':
        # Create timestamped filename to avoid overwriting
        filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(f"# Research Report\n\n")
            f.write(f"**Query:** {query}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Agent:** Tavily-Powered Deep Research Agent\n\n")
            f.write(f"**Search Depth:** {agent.search_depth}\n\n")
            f.write(f"**Sources:** {result['sources']}\n\n")
            f.write(f"**Citations:** {result['citations']}\n\n")
            f.write(f"**Duration:** {result['duration_seconds']:.2f} seconds\n\n")
            f.write(f"---\n\n")
            f.write(result["report"])
        print(f"✅ Saved to {filename}")


if __name__ == "__main__":
    # Entry point for script execution
    # Runs the interactive demo for Tavily-powered research
    asyncio.run(main())

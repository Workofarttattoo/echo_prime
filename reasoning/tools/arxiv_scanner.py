import requests
import feedparser
import json
from urllib.parse import urlencode
from typing import List, Dict, Any, Optional
from mcp_server.registry import ToolRegistry

class ArxivScanner:
    """
    Tool to ingest scientific papers from arXiv and other preprint servers.
    """
    PLATFORMS = {
        'arxiv': 'http://export.arxiv.org/api/query',
        'biorxiv': 'https://api.biorxiv.org/details/biorxiv',
        'medrxiv': 'https://api.biorxiv.org/details/medrxiv',
    }

    @ToolRegistry.register(name="scan_arxiv")
    def scan(self, query: str = "quantum computing", max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Scans arXiv for papers matching the query.
        Returns a list of structured paper objects.
        """
        print(f"ARXIV: Scanning for '{query}'...")
        results = []
        
        # 1. arXiv
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            url = f"{self.PLATFORMS['arxiv']}?{urlencode(params)}"
            feed = feedparser.parse(url)
            
            for e in feed.entries:
                results.append({
                    'title': e.title,
                    'authors': [a.name for a in e.authors],
                    'summary': e.summary.replace('\n', ' '),
                    'url': e.link,
                    'published': e.published,
                    'source': 'arXiv'
                })
        except Exception as e:
            print(f"ARXIV ERROR: {e}")
            
        return results

if __name__ == "__main__":
    scanner = ArxivScanner()
    papers = scanner.scan("cancer metabolism", 2)
    print(json.dumps(papers, indent=2))


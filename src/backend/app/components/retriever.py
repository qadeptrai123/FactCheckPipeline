from typing import List, Dict, Any

class DocumentRetriever:
    def retrieve(self, query: str, method: str, top_k: int) -> List[Dict[str, Any]]:
        # MOCK IMPLEMENTATION
        return [
            {"id": i, "content": f"Mock document {i} for query '{query}' using {method}", "score": 0.99 - (i * 0.01)}
            for i in range(1, top_k + 1)
        ]

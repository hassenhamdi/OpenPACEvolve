"""
Failure Memory for Hierarchical Context Management.

Persists pruned failures and rejected hypotheses to prevent rediscovery.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Record of a failed approach."""
    id: str
    description: str
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    idea_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    code_hash: Optional[str] = None


class FailureMemory:
    """
    Persistent record of pruned failures and rejected hypotheses.
    
    Prevents rediscovery of known bad approaches by:
    - Recording failed ideas and hypotheses
    - Checking new proposals against failure history
    - Providing failure context to the LLM
    """
    
    def __init__(self, max_failures: int = 100):
        """
        Initialize failure memory.
        
        Args:
            max_failures: Maximum failures to remember.
        """
        self.max_failures = max_failures
        self.failures: List[FailureRecord] = []
        self._description_hashes: Set[str] = set()
        self._code_hashes: Set[str] = set()
    
    def _hash_description(self, description: str) -> str:
        """Create hash of description for quick lookup."""
        import hashlib
        normalized = description.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _hash_code(self, code: str) -> str:
        """Create hash of code for quick lookup."""
        import hashlib
        # Normalize whitespace
        normalized = " ".join(code.split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def record_failure(
        self,
        description: str,
        reason: str,
        metrics: Optional[Dict[str, float]] = None,
        idea_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        """
        Record a failed approach.
        
        Args:
            description: Description of the failed approach.
            reason: Why it failed.
            metrics: Performance metrics.
            idea_id: Associated idea ID if any.
            hypothesis_id: Associated hypothesis ID if any.
            code: Code that failed if available.
        """
        import uuid
        
        desc_hash = self._hash_description(description)
        code_hash = self._hash_code(code) if code else None
        
        # Check if already recorded
        if desc_hash in self._description_hashes:
            logger.debug(f"Failure already recorded: {description[:50]}...")
            return
        
        failure = FailureRecord(
            id=str(uuid.uuid4()),
            description=description,
            reason=reason,
            metrics=metrics or {},
            idea_id=idea_id,
            hypothesis_id=hypothesis_id,
            code_hash=code_hash,
        )
        
        self.failures.append(failure)
        self._description_hashes.add(desc_hash)
        if code_hash:
            self._code_hashes.add(code_hash)
        
        # Prune old failures if over limit
        if len(self.failures) > self.max_failures:
            removed = self.failures.pop(0)
            self._description_hashes.discard(self._hash_description(removed.description))
            if removed.code_hash:
                self._code_hashes.discard(removed.code_hash)
        
        logger.info(f"Recorded failure: {description[:50]}... Reason: {reason}")
    
    def is_known_failure(self, description: str) -> bool:
        """Check if a description matches a known failure."""
        desc_hash = self._hash_description(description)
        return desc_hash in self._description_hashes
    
    def is_code_failed(self, code: str) -> bool:
        """Check if code matches a known failed implementation."""
        code_hash = self._hash_code(code)
        return code_hash in self._code_hashes
    
    def get_similar_failures(self, description: str, max_results: int = 3) -> List[FailureRecord]:
        """
        Get failures similar to the given description.
        
        Uses simple keyword matching (could be enhanced with embeddings).
        """
        keywords = set(description.lower().split())
        scored = []
        
        for failure in self.failures:
            failure_keywords = set(failure.description.lower().split())
            overlap = len(keywords & failure_keywords)
            if overlap > 0:
                scored.append((overlap, failure))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:max_results]]
    
    def get_failure_summary(self, max_failures: int = 5) -> str:
        """Get a summary of recent failures for context."""
        if not self.failures:
            return "No recorded failures yet."
        
        recent = self.failures[-max_failures:]
        lines = ["## Known Failed Approaches (Avoid These)"]
        
        for failure in recent:
            lines.append(f"- {failure.description}: {failure.reason}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "failures": [
                {
                    "id": f.id,
                    "description": f.description,
                    "reason": f.reason,
                    "metrics": f.metrics,
                    "timestamp": f.timestamp,
                    "idea_id": f.idea_id,
                    "hypothesis_id": f.hypothesis_id,
                    "code_hash": f.code_hash,
                }
                for f in self.failures
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureMemory":
        """Deserialize from dictionary."""
        memory = cls()
        
        for f_data in data.get("failures", []):
            failure = FailureRecord(
                id=f_data["id"],
                description=f_data["description"],
                reason=f_data["reason"],
                metrics=f_data.get("metrics", {}),
                timestamp=f_data.get("timestamp", time.time()),
                idea_id=f_data.get("idea_id"),
                hypothesis_id=f_data.get("hypothesis_id"),
                code_hash=f_data.get("code_hash"),
            )
            memory.failures.append(failure)
            memory._description_hashes.add(memory._hash_description(failure.description))
            if failure.code_hash:
                memory._code_hashes.add(failure.code_hash)
        
        return memory

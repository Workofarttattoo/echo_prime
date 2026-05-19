"""
Persistent Memory System for Echo Lite

Features:
- SQLite-based storage (lightweight, embedded)
- Episodic memory (experiences)
- Semantic memory (knowledge)
- Identity persistence (personality, state)
- Automatic save/load
- Memory consolidation
"""

import sqlite3
import json
import time
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class MemoryEntry:
    """Single memory entry"""
    timestamp: float
    memory_type: str  # "episodic", "semantic", "identity"
    content: str
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'memory_type': self.memory_type,
            'content': self.content,
            'embedding': json.dumps(self.embedding) if self.embedding else None,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': json.dumps(self.metadata) if self.metadata else None
        }


class PersistentMemory:
    """
    Persistent memory system using SQLite

    Survives reboots and maintains continuous identity
    """

    def __init__(self, db_path: str = "echo_lite_memory.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None

        # Thread safety lock for all database operations
        self._db_lock = threading.Lock()

        # Initialize database
        self._init_database()

        # Load identity
        self.identity = self._load_identity()

        print(f"💾 Persistent memory initialized: {db_path}")
        print(f"   Identity: {self.identity.get('name', 'Echo Lite')}")
        print(f"   Total memories: {self.count_memories()}")

    def _init_database(self):
        """Initialize SQLite database"""
        # check_same_thread=False needed for multi-threaded agent access
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # Note: Don't store a shared cursor - create per-operation for thread safety
        # Create tables using a local cursor
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS identity (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cognitive_state (
                timestamp REAL PRIMARY KEY,
                state_vector TEXT NOT NULL,
                cycle_count INTEGER,
                metadata TEXT
            )
        ''')

        # Create indices for fast retrieval
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp DESC)
        ''')

        self.conn.commit()

    def _load_identity(self) -> Dict[str, Any]:
        """Load identity from database"""
        self.cursor.execute('SELECT key, value FROM identity')
        rows = self.cursor.fetchall()

        identity = {}
        for key, value in rows:
            try:
                identity[key] = json.loads(value)
            except:
                identity[key] = value

        # Default identity if new
        if not identity:
            identity = {
                'name': 'Echo Lite',
                'birth_timestamp': time.time(),
                'version': '1.0',
                'personality': {
                    'curious': 0.8,
                    'helpful': 0.9,
                    'analytical': 0.7
                },
                'total_cycles': 0,
                'total_memories': 0
            }
            self._save_identity(identity)

        return identity

    def _save_identity(self, identity: Dict[str, Any]):
        """Save identity to database"""
        for key, value in identity.items():
            value_str = json.dumps(value) if not isinstance(value, str) else value
            self.cursor.execute('''
                INSERT OR REPLACE INTO identity (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value_str))
        self.conn.commit()

    def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a new memory

        Args:
            content: Memory content
            memory_type: "episodic", "semantic", or "identity"
            importance: 0-1 importance score
            embedding: Optional vector embedding
            metadata: Optional metadata dict

        Returns:
            Memory ID
        """
        entry = MemoryEntry(
            timestamp=time.time(),
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            metadata=metadata
        )

        data = entry.to_dict()

        with self._db_lock:
            self.cursor.execute('''
                INSERT INTO memories (
                    timestamp, memory_type, content, embedding,
                    importance, access_count, last_accessed, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['memory_type'],
                data['content'],
                data['embedding'],
                data['importance'],
                0,
                None,
                data['metadata']
            ))

            self.conn.commit()

            # Note: Identity total_memories is updated periodically in agent_runtime,
            # not on every write to avoid unnecessary I/O overhead

            return self.cursor.lastrowid

    def recall_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Recall memories from storage

        Args:
            memory_type: Filter by type (None = all)
            limit: Maximum number to return
            min_importance: Minimum importance threshold

        Returns:
            List of memory dictionaries
        """
        query = '''
            SELECT id, timestamp, memory_type, content, importance,
                   access_count, last_accessed, metadata
            FROM memories
            WHERE importance >= ?
        '''
        params = [min_importance]

        if memory_type:
            query += ' AND memory_type = ?'
            params.append(memory_type)

        query += ' ORDER BY importance DESC, timestamp DESC LIMIT ?'
        params.append(limit)

        with self._db_lock:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            memories = []
            for row in rows:
                memory = {
                    'id': row[0],
                    'timestamp': row[1],
                    'memory_type': row[2],
                    'content': row[3],
                    'importance': row[4],
                    'access_count': row[5],
                    'last_accessed': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                memories.append(memory)

                # Update access count
                self.cursor.execute('''
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE id = ?
                ''', (time.time(), row[0]))

            self.conn.commit()
            return memories

    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by content (simple text search)

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching memories
        """
        self.cursor.execute('''
            SELECT id, timestamp, memory_type, content, importance,
                   access_count, metadata
            FROM memories
            WHERE content LIKE ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        rows = self.cursor.fetchall()

        memories = []
        for row in rows:
            memory = {
                'id': row[0],
                'timestamp': row[1],
                'memory_type': row[2],
                'content': row[3],
                'importance': row[4],
                'access_count': row[5],
                'metadata': json.loads(row[6]) if row[6] else None
            }
            memories.append(memory)

        return memories

    def save_cognitive_state(
        self,
        state_vector: np.ndarray,
        cycle_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save current cognitive state"""
        # Handle both numpy arrays and lists (from cognitive_cycle)
        if isinstance(state_vector, np.ndarray):
            state_json = json.dumps(state_vector.tolist())
        else:
            state_json = json.dumps(state_vector)
        metadata_json = json.dumps(metadata) if metadata else None

        with self._db_lock:
            self.cursor.execute('''
                INSERT INTO cognitive_state (timestamp, state_vector, cycle_count, metadata)
                VALUES (?, ?, ?, ?)
            ''', (time.time(), state_json, cycle_count, metadata_json))

            self.conn.commit()

    def load_latest_cognitive_state(self) -> Optional[Dict[str, Any]]:
        """Load most recent cognitive state"""
        self.cursor.execute('''
            SELECT timestamp, state_vector, cycle_count, metadata
            FROM cognitive_state
            ORDER BY timestamp DESC
            LIMIT 1
        ''')

        row = self.cursor.fetchone()
        if not row:
            return None

        return {
            'timestamp': row[0],
            'state_vector': np.array(json.loads(row[1])),
            'cycle_count': row[2],
            'metadata': json.loads(row[3]) if row[3] else None
        }

    def update_identity(self, updates: Dict[str, Any]):
        """Update identity attributes"""
        self.identity.update(updates)
        self._save_identity(self.identity)

    def get_identity(self) -> Dict[str, Any]:
        """Get current identity"""
        return self.identity.copy()

    def count_memories(self, memory_type: Optional[str] = None) -> int:
        """Count total memories"""
        with self._db_lock:
            if memory_type:
                self.cursor.execute(
                    'SELECT COUNT(*) FROM memories WHERE memory_type = ?',
                    (memory_type,)
                )
            else:
                self.cursor.execute('SELECT COUNT(*) FROM memories')

            return self.cursor.fetchone()[0]

    def consolidate_memories(self, days_old: int = 7):
        """
        Memory consolidation - reduce importance of old, rarely accessed memories

        Mimics sleep consolidation in biological systems
        """
        cutoff_time = time.time() - (days_old * 24 * 3600)

        self.cursor.execute('''
            UPDATE memories
            SET importance = importance * 0.9
            WHERE timestamp < ?
            AND access_count < 3
            AND memory_type = 'episodic'
        ''', (cutoff_time,))

        affected = self.cursor.rowcount
        self.conn.commit()

        print(f"🧹 Consolidated {affected} old memories")
        return affected

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'total_memories': self.count_memories(),
            'episodic': self.count_memories('episodic'),
            'semantic': self.count_memories('semantic'),
            'identity_memories': self.count_memories('identity')
        }

        # Most accessed
        self.cursor.execute('''
            SELECT content, access_count
            FROM memories
            ORDER BY access_count DESC
            LIMIT 1
        ''')
        row = self.cursor.fetchone()
        if row:
            stats['most_accessed'] = {
                'content': row[0][:50] + '...',
                'count': row[1]
            }

        # Oldest memory
        self.cursor.execute('''
            SELECT content, timestamp
            FROM memories
            ORDER BY timestamp ASC
            LIMIT 1
        ''')
        row = self.cursor.fetchone()
        if row:
            stats['oldest_memory'] = {
                'content': row[0][:50] + '...',
                'age_hours': (time.time() - row[1]) / 3600
            }

        return stats

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    # Demo
    print("💾 Persistent Memory System Demo\n")

    memory = PersistentMemory("test_memory.db")

    # Store some memories
    memory.store_memory(
        "I learned about quantum computing today",
        memory_type="episodic",
        importance=0.8,
        metadata={'topic': 'quantum', 'mood': 'curious'}
    )

    memory.store_memory(
        "Quantum entanglement is a physical phenomenon",
        memory_type="semantic",
        importance=0.9
    )

    # Recall memories
    print("📖 Recent memories:")
    memories = memory.recall_memories(limit=5)
    for mem in memories:
        print(f"  - [{mem['memory_type']}] {mem['content'][:60]}...")

    # Search
    print("\n🔍 Search 'quantum':")
    results = memory.search_memories("quantum")
    for mem in results:
        print(f"  - {mem['content'][:60]}...")

    # Statistics
    print("\n📊 Statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Identity
    print(f"\n🎭 Identity:")
    identity = memory.get_identity()
    print(f"  Name: {identity['name']}")
    print(f"  Total cycles: {identity['total_cycles']}")
    print(f"  Total memories: {identity['total_memories']}")

    memory.close()

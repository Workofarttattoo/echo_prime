import numpy as np
import os
import json
import time
import torch  # pyright: ignore[reportUnusedImport]
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from collections import deque  # pyright: ignore[reportUnusedImport]
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using simplified memory indexing.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: Sentence transformers not available. Using simplified embeddings.")


class WorkingMemory:
    """
    Simulates Prefrontal Cortex working memory.
    Capacity: 7 ± 2 chunks.
    Enhanced with importance-based retention.
    """
    def __init__(self, capacity: int = 7, chunk_dim: int = 1000):
        self.capacity = capacity
        self.chunk_dim = chunk_dim
        self.chunks = []  # List of (vector, timestamp, importance)
        self.decay_rate = 0.05  # Per step decay of 'calcium' levels
        self.importance_threshold = 0.5
    
    def store(self, vector: np.ndarray, importance: float = 0.5):
        """Store vector with importance score"""
        timestamp = time.time()
        
        if len(self.chunks) >= self.capacity:
            # Evict least important or oldest chunk
            if importance > self.importance_threshold:
                # Find least important chunk
                min_importance_idx = min(
                    range(len(self.chunks)),
                    key=lambda i: self.chunks[i][2]
                )
                self.chunks.pop(min_importance_idx)
            else:
                # Evict oldest
                self.chunks.pop(0)
        
        self.chunks.append((vector, timestamp, importance))
    
    def retrieve_all(self) -> List[np.ndarray]:
        """Retrieve all chunks, sorted by importance"""
        sorted_chunks = sorted(self.chunks, key=lambda x: x[2], reverse=True)
        return [chunk[0] for chunk in sorted_chunks]
    
    def update_importance(self, idx: int, new_importance: float):
        """Update importance of a chunk"""
        if 0 <= idx < len(self.chunks):
            vector, timestamp, _ = self.chunks[idx]
            self.chunks[idx] = (vector, timestamp, new_importance)


class EpisodicMemory:
    """
    Enhanced episodic memory with vector database indexing.
    Uses FAISS for efficient similarity search.
    """
    def __init__(self, feature_dim: int = 1024, use_vector_db: bool = True):
        self.feature_dim = feature_dim
        self.storage = []  # List of stored episodes
        self.use_vector_db = use_vector_db
        
        if use_vector_db and FAISS_AVAILABLE:
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(feature_dim)
            self.episode_metadata = []  # Store metadata for each episode
        else:
            # Fallback: LSH Projection matrix
            self.lsh_projection = np.random.randn(256, feature_dim)
    
    def _get_hash(self, x: np.ndarray) -> str:
        """LSH hash for fallback mode"""
        h = np.sign(self.lsh_projection @ x)
        return "".join(['1' if val > 0 else '0' for val in h])
    
    def store_episode(self, feature_vector: np.ndarray, metadata: Optional[Dict] = None):
        """Store episode with optional metadata"""
        # Ensure correct dimension
        if len(feature_vector) != self.feature_dim:
            # Pad or truncate
            if len(feature_vector) < self.feature_dim:
                feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)))
            else:
                feature_vector = feature_vector[:self.feature_dim]
        
        # Normalize
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
        
        self.storage.append(feature_vector)
        
        if self.use_vector_db and FAISS_AVAILABLE:
            # Add to FAISS index
            feature_vector_reshaped = feature_vector.reshape(1, -1).astype('float32')
            self.index.add(feature_vector_reshaped)
            self.episode_metadata.append(metadata or {})
        else:
            self.episode_metadata.append(metadata or {})
    
    def retrieve_nearest(self, query_vector: np.ndarray, k: int = 1) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve k nearest episodes"""
        if not self.storage:
            return np.zeros(self.feature_dim), []
        
        # Normalize query
        norm_q = np.linalg.norm(query_vector)
        if norm_q < 1e-9:
            return np.zeros(self.feature_dim), []
        
        query_vector = query_vector / norm_q
        
        if self.use_vector_db and FAISS_AVAILABLE:
            # Use FAISS for efficient search
            query_reshaped = query_vector.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_reshaped, min(k, len(self.storage)))
            
            results = []
            for idx in indices[0]:
                if idx < len(self.storage):
                    results.append((self.storage[idx], self.episode_metadata[idx] if idx < len(self.episode_metadata) else {}))
            
            if results:
                return results[0][0], [r[1] for r in results]
            return np.zeros(self.feature_dim), []
        else:
            # Fallback: cosine similarity
            similarities = [np.dot(query_vector, x) / (np.linalg.norm(x) + 1e-9) for x in self.storage]
            best_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in best_indices:
                results.append((self.storage[idx], {}))
            
            if results:
                return results[0][0], [r[1] for r in results]
            return np.zeros(self.feature_dim), []
    
    def consolidate(self, replay_ratio: float = 0.1):
        """
        Memory consolidation: replay and strengthen important memories.
        Simulates sleep-like consolidation.
        """
        if not self.storage:
            return
        
        # Select subset for replay
        num_replay = max(1, int(len(self.storage) * replay_ratio))
        replay_indices = np.random.choice(len(self.storage), num_replay, replace=False)
        
        # Replay and strengthen (simplified: just re-index)
        for idx in replay_indices:
            episode = self.storage[idx]
            # Re-add to index (strengthens connection)
            if self.use_vector_db and FAISS_AVAILABLE:
                episode_reshaped = episode.reshape(1, -1).astype('float32')
                self.index.add(episode_reshaped)
    
    def compress(self, compression_ratio: float = 0.5):
        """
        Compress memory by removing less important episodes.
        """
        if len(self.storage) == 0:
            return
        
        # Simple compression: keep most recent and most similar to others
        num_keep = max(1, int(len(self.storage) * (1 - compression_ratio)))
        
        # Keep most recent
        recent_indices = list(range(len(self.storage) - num_keep, len(self.storage)))
        
        # Rebuild storage and index
        new_storage = [self.storage[i] for i in recent_indices]
        new_metadata = [self.episode_metadata[i] for i in recent_indices if i < len(self.episode_metadata)]
        
        self.storage = new_storage
        
        if self.use_vector_db and FAISS_AVAILABLE:
            # Rebuild index
            self.index.reset()
            if self.storage:
                vectors = np.array(self.storage).astype('float32')
                self.index.add(vectors)
            self.episode_metadata = new_metadata
        else:
            self.episode_metadata = new_metadata
    
    def save(self, path: str):
        """Saves episodic storage to a numpy file."""
        if self.storage:
            np.save(path, np.array(self.storage))
            # Save metadata
            metadata_path = path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.episode_metadata, f)
    
    def load(self, path: str):
        """Loads episodic storage from a numpy file."""
        if os.path.exists(path):
            try:
                # Use allow_pickle=True for complex object arrays
                loaded_data = np.load(path, allow_pickle=True)
                if loaded_data.size > 0:
                    self.storage = list(loaded_data)
                else:
                    self.storage = []
            except Exception as e:
                print(f"Warning: Could not load episodic memory from {path}: {e}")
                self.storage = []
            
            # Load metadata
            metadata_path = path.replace('.npy', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.episode_metadata = json.load(f)
            else:
                self.episode_metadata = [{}] * len(self.storage)
            
            # Rebuild index
            if self.use_vector_db and FAISS_AVAILABLE and self.storage:
                self.index.reset()
                vectors = np.array(self.storage).astype('float32')
                self.index.add(vectors)


class SemanticMemory:
    """
    Enhanced semantic memory with learned embeddings.
    Uses SentenceTransformer for better semantic representations.
    """
    def __init__(self, dimension: int = 384, use_embeddings: bool = True):
        self.dimension = dimension
        self.knowledge_base = {}
        self.use_embeddings = use_embeddings
        
        if use_embeddings:
            try:
                model_id = 'sentence-transformers/all-MiniLM-L6-v2'
                # Explicitly disable auth token to avoid issues with expired system tokens
                self.embedder = SentenceTransformer(model_id, use_auth_token=False)
                self.dimension = self.embedder.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer: {e}")
                self.use_embeddings = False
                self.embedder = None
    
    def _embed(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if self.use_embeddings and self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # Fallback: random embedding
            return np.random.randn(self.dimension)
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution via FFT: F^-1(F(a) * F(b))"""
        # Ensure same dimension
        min_dim = min(len(a), len(b))
        a = a[:min_dim]
        b = b[:min_dim]
        
        fa = np.fft.fft(a)
        fb = np.fft.fft(b)
        result = np.fft.ifft(fa * fb).real
        
        # Pad back to original dimension if needed
        if len(result) < self.dimension:
            result = np.pad(result, (0, self.dimension - len(result)))
        
        return result[:self.dimension]
    
    def unbind(self, bound: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Approximate unbinding using involution (approximate inverse)."""
        min_dim = min(len(bound), len(a))
        bound = bound[:min_dim]
        a = a[:min_dim]
        
        inv_a = np.roll(a[::-1], 1)
        result = self.bind(bound, inv_a)
        
        if len(result) < self.dimension:
            result = np.pad(result, (0, self.dimension - len(result)))
        
        return result[:self.dimension]
    
    def store_concept(self, name: str, vector: Optional[np.ndarray] = None, text: Optional[str] = None):
        """Store concept with vector or text"""
        if vector is not None:
            # Ensure correct dimension
            if len(vector) != self.dimension:
                if len(vector) < self.dimension:
                    vector = np.pad(vector, (0, self.dimension - len(vector)))
                else:
                    vector = vector[:self.dimension]
            self.knowledge_base[name] = vector
        elif text is not None:
            # Generate embedding from text
            embedding = self._embed(text)
            self.knowledge_base[name] = embedding
        else:
            raise ValueError("Either vector or text must be provided")
    
    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve similar concepts using embeddings"""
        if not self.knowledge_base:
            return []
        
        query_embedding = self._embed(query)
        
        similarities = []
        for name, vector in self.knowledge_base.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vector) + 1e-9
            )
            similarities.append((name, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, path: str):
        """Saves semantic knowledge base to a JSON-compatible format."""
        serializable = {k: v.tolist() for k, v in self.knowledge_base.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)
    
    def load(self, path: str):
        """Loads semantic knowledge base from JSON."""
        if os.path.exists(path):
            try:
                if os.path.getsize(path) < 2: # {} is at least 2 bytes
                    self.knowledge_base = {}
                    return
                with open(path, "r") as f:
                    content = f.read().strip()
                    if not content:
                        self.knowledge_base = {}
                        return
                    data = json.loads(content)
                self.knowledge_base = {k: np.array(v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not load semantic memory from {path}: {e}")
                self.knowledge_base = {}


class MemoryConsolidation:
    """
    Implements sleep-like memory consolidation.
    Replays and strengthens important memories.
    """
    def __init__(self, episodic_memory: EpisodicMemory, semantic_memory: SemanticMemory):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.consolidation_interval = 1000  # Consolidate every N steps
        self.step_count = 0
    
    def consolidate_step(self):
        """Perform one consolidation step"""
        self.step_count += 1
        
        if self.step_count % self.consolidation_interval == 0:
            # Replay episodic memories
            self.episodic.consolidate(replay_ratio=0.1)
            
            # Transfer important episodic memories to semantic
            # (Simplified: just mark for future transfer)
    
    def transfer_to_semantic(self, importance_threshold: float = 0.7):
        """
        Transfer important episodic memories to semantic memory.
        """
        # This would analyze episodic memories and extract semantic concepts
        # For now, it's a placeholder
        pass


class AdaptiveForgetting:
    """
    Implements adaptive forgetting based on importance and recency.
    """
    def __init__(self, decay_rate: float = 0.01, importance_weight: float = 0.5):
        self.decay_rate = decay_rate
        self.importance_weight = importance_weight
    
    def compute_forgetting_probability(self, age: float, importance: float) -> float:
        """
        Compute probability of forgetting based on age and importance.
        """
        # Older and less important memories are more likely to be forgotten
        age_factor = 1 - np.exp(-self.decay_rate * age)
        importance_factor = 1 - importance
        
        forgetting_prob = self.importance_weight * age_factor + (1 - self.importance_weight) * importance_factor
        
        return min(1.0, max(0.0, forgetting_prob))
    
    def should_forget(self, age: float, importance: float) -> bool:
        """Determine if memory should be forgotten"""
        prob = self.compute_forgetting_probability(age, importance)
        return np.random.random() < prob


class MemoryManager:
    """
    Enhanced orchestrator for all memory systems with vector databases and consolidation.
    """
    def __init__(self, data_dir: str = "memory_data", use_vector_db: bool = True):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(use_vector_db=use_vector_db)
        self.semantic = SemanticMemory(use_embeddings=True)
        self.consolidation = MemoryConsolidation(self.episodic, self.semantic)
        self.forgetting = AdaptiveForgetting()
        
        # Paths
        self.episodic_path = os.path.join(data_dir, "episodic.npy")
        self.semantic_path = os.path.join(data_dir, "semantic.json")
        
        # Auto-load
        self.episodic.load(self.episodic_path)
        self.semantic.load(self.semantic_path)
        
        # Consolidation tracking
        self.last_consolidation = time.time()
        self.consolidation_interval = 3600  # 1 hour
    
    def process_input(self, input_vector: np.ndarray, importance: float = 0.5):
        """Process input with importance weighting"""
        # 1. Update Working Memory
        self.working.store(input_vector, importance)
        
        # 2. Store to Episodic with metadata
        metadata = {
            "timestamp": time.time(),
            "importance": importance
        }
        self.episodic.store_episode(input_vector, metadata)
        
        # 3. Periodic consolidation
        current_time = time.time()
        if current_time - self.last_consolidation > self.consolidation_interval:
            self.consolidation.consolidate_step()
            self.last_consolidation = current_time
        
        # 4. Adaptive forgetting (remove old, unimportant memories)
        self._apply_forgetting()
        
        # 5. Auto-save
        self.episodic.save(self.episodic_path)
        self.semantic.save(self.semantic_path)
    
    def _apply_forgetting(self):
        """Apply adaptive forgetting to episodic memory"""
        if not self.episodic.storage:
            return
        
        current_time = time.time()
        to_remove = []
        
        for i, metadata in enumerate(self.episodic.episode_metadata):
            if i >= len(self.episodic.storage):
                break
            
            age = current_time - metadata.get("timestamp", current_time)
            importance = metadata.get("importance", 0.5)
            
            if self.forgetting.should_forget(age, importance):
                to_remove.append(i)
        
        # Remove forgotten memories (in reverse order to maintain indices)
        for idx in reversed(to_remove):
            if idx < len(self.episodic.storage):
                self.episodic.storage.pop(idx)
                if idx < len(self.episodic.episode_metadata):
                    self.episodic.episode_metadata.pop(idx)
        
        # Rebuild index if using vector DB
        if self.episodic.use_vector_db and FAISS_AVAILABLE and self.episodic.storage:
            self.episodic.index.reset()
            vectors = np.array(self.episodic.storage).astype('float32')
            self.episodic.index.add(vectors)
    
    def consolidate_now(self):
        """Force immediate consolidation"""
        self.consolidation.consolidate_step()
        self.last_consolidation = time.time()
    
    def compress_memory(self, ratio: float = 0.3):
        """Compress memory by removing less important episodes"""
        self.episodic.compress(compression_ratio=ratio)

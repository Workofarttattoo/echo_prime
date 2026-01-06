#!/usr/bin/env python3
"""
ECH0-PRIME Multi-Modal Sensory Integration System
Advanced fusion of vision, audio, and text inputs with cross-modal reasoning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import cv2
import librosa
from collections import deque
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model
import speech_recognition as sr
from PIL import Image
import io
import base64
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Modality(Enum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"


class FusionStrategy(Enum):
    EARLY_FUSION = "early_fusion"      # Fuse at feature level
    LATE_FUSION = "late_fusion"        # Fuse at decision level
    HYBRID_FUSION = "hybrid_fusion"    # Multi-level fusion
    ATTENTION_FUSION = "attention_fusion"  # Attention-based fusion
    TRANSFORMER_FUSION = "transformer_fusion"  # Transformer-based fusion


@dataclass
class ModalityData:
    """Container for multi-modal data"""
    modality: Modality
    data: Any
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_features: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class FusedRepresentation:
    """Fused multi-modal representation"""
    fused_features: np.ndarray
    modality_contributions: Dict[Modality, float]
    fusion_confidence: float
    attention_weights: Optional[np.ndarray] = None
    cross_modal_relations: Dict[Tuple[Modality, Modality], float] = field(default_factory=dict)


@dataclass
class CrossModalAnalogy:
    """Cross-modal analogy representation"""
    source_modality: Modality
    target_modality: Modality
    source_concept: str
    target_concept: str
    analogy_strength: float
    mapping_explanation: str
    confidence: float


class ModalityEncoder(nn.Module):
    """
    Encoder for individual modality processing
    """

    def __init__(self, modality: Modality, feature_dim: int = 512):
        super(ModalityEncoder, self).__init__()
        self.modality = modality
        self.feature_dim = feature_dim

        if modality == Modality.VISION:
            # Vision encoder (simplified CLIP-like)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(256 * 16, feature_dim)
            )
        elif modality == Modality.AUDIO:
            # Audio encoder (simplified Wav2Vec-like)
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.AdaptiveAvgPool1d(128),
                nn.Flatten(),
                nn.Linear(128 * 128, feature_dim)
            )
        elif modality == Modality.TEXT:
            # Text encoder (simplified BERT-like)
            self.embedding = nn.Embedding(30000, 256)  # Vocabulary size, embedding dim
            self.encoder = nn.Sequential(
                nn.Linear(256, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            # Generic encoder
            self.encoder = nn.Linear(256, feature_dim)

    def forward(self, x):
        if self.modality == Modality.TEXT:
            # Handle text input (token indices)
            embedded = self.embedding(x)
            # Simple pooling
            pooled = torch.mean(embedded, dim=1)
            return self.encoder(pooled)
        else:
            return self.encoder(x)


class MultiModalFusionNetwork(nn.Module):
    """
    Neural network for multi-modal fusion
    """

    def __init__(self, modalities: List[Modality], feature_dim: int = 512,
                 fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION):
        super(MultiModalFusionNetwork, self).__init__()
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.fusion_strategy = fusion_strategy

        # Modality-specific encoders
        self.encoders = nn.ModuleDict({
            modality.value: ModalityEncoder(modality, feature_dim)
            for modality in modalities
        })

        if fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            # Attention-based fusion
            self.attention_weights = nn.Linear(feature_dim * len(modalities), len(modalities))
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * len(modalities), feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
        elif fusion_strategy == FusionStrategy.TRANSFORMER_FUSION:
            # Transformer-based fusion (simplified)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=8, dim_feedforward=2048
            )
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
            self.fusion_projection = nn.Linear(feature_dim, feature_dim)
        else:
            # Simple concatenation fusion
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * len(modalities), feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )

        # Cross-modal relation modeling
        self.cross_modal_predictor = nn.Bilinear(feature_dim, feature_dim, feature_dim)

        # Modality confidence predictor
        self.confidence_predictor = nn.Linear(feature_dim, len(modalities))

    def forward(self, modality_inputs: Dict[Modality, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through fusion network"""
        # Encode each modality
        modality_features = {}
        for modality, input_tensor in modality_inputs.items():
            modality_features[modality] = self.encoders[modality.value](input_tensor)

        # Apply fusion strategy
        if self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            fused_features, attention_weights = self._attention_fusion(modality_features)
        elif self.fusion_strategy == FusionStrategy.TRANSFORMER_FUSION:
            fused_features, attention_weights = self._transformer_fusion(modality_features)
        else:
            fused_features, attention_weights = self._concatenation_fusion(modality_features)

        # Predict modality confidences
        modality_confidences = torch.sigmoid(self.confidence_predictor(fused_features))

        # Predict cross-modal relations
        cross_modal_relations = self._predict_cross_modal_relations(modality_features)

        return {
            'fused_features': fused_features,
            'attention_weights': attention_weights,
            'modality_confidences': modality_confidences,
            'cross_modal_relations': cross_modal_relations,
            'modality_features': modality_features
        }

    def _attention_fusion(self, modality_features: Dict[Modality, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention-based fusion"""
        # Concatenate all modality features
        concat_features = torch.cat(list(modality_features.values()), dim=-1)

        # Compute attention weights
        attention_logits = self.attention_weights(concat_features)
        attention_weights = F.softmax(attention_logits, dim=-1)

        # Apply attention to modality features
        weighted_features = []
        for i, modality in enumerate(self.modalities):
            weight = attention_weights[:, i:i+1]
            weighted_feature = modality_features[modality] * weight
            weighted_features.append(weighted_feature)

        # Fuse weighted features
        fused = torch.cat(weighted_features, dim=-1)
        fused = self.fusion_layer(fused)

        return fused, attention_weights

    def _transformer_fusion(self, modality_features: Dict[Modality, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformer-based fusion"""
        # Stack modality features
        stacked_features = torch.stack(list(modality_features.values()), dim=0)  # [num_modalities, batch, feature_dim]

        # Apply transformer
        transformer_output = self.transformer(stacked_features)

        # Pool across modalities (simple average)
        fused = torch.mean(transformer_output, dim=0)
        fused = self.fusion_projection(fused)

        # Compute attention weights (simplified)
        attention_weights = torch.ones(len(self.modalities)) / len(self.modalities)

        return fused, attention_weights

    def _concatenation_fusion(self, modality_features: Dict[Modality, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple concatenation fusion"""
        concat_features = torch.cat(list(modality_features.values()), dim=-1)
        fused = self.fusion_layer(concat_features)

        # Equal attention weights
        attention_weights = torch.ones(len(self.modalities)) / len(self.modalities)

        return fused, attention_weights

    def _predict_cross_modal_relations(self, modality_features: Dict[Modality, torch.Tensor]) -> torch.Tensor:
        """Predict relations between modalities"""
        # Compute pairwise relations
        modalities_list = list(self.modalities)
        relations = []

        for i, mod1 in enumerate(modalities_list):
            for j, mod2 in enumerate(modalities_list):
                if i != j:
                    relation = self.cross_modal_predictor(
                        modality_features[mod1],
                        modality_features[mod2]
                    )
                    relations.append(relation)

        # Stack relations
        if relations:
            return torch.stack(relations, dim=1)
        else:
            return torch.empty(0)


class SensoryDataCompressor:
    """
    Compresses and efficiently stores sensory data
    """

    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.compressed_data_store = {}
        self.compression_stats = {}

    def compress_sensory_data(self, modality_data: ModalityData) -> Dict[str, Any]:
        """Compress sensory data for efficient storage"""
        modality = modality_data.modality

        if modality == Modality.VISION:
            compressed = self._compress_vision_data(modality_data.data)
        elif modality == Modality.AUDIO:
            compressed = self._compress_audio_data(modality_data.data)
        elif modality == Modality.TEXT:
            compressed = self._compress_text_data(modality_data.data)
        else:
            compressed = self._compress_generic_data(modality_data.data)

        # Store compressed data
        data_id = f"{modality.value}_{int(time.time() * 1000)}"
        self.compressed_data_store[data_id] = {
            'original_size': self._estimate_data_size(modality_data.data),
            'compressed_size': len(compressed) if isinstance(compressed, (bytes, str)) else compressed.nbytes,
            'compression_ratio': self.compression_ratio,
            'data': compressed,
            'metadata': modality_data.metadata,
            'timestamp': modality_data.timestamp
        }

        return {
            'data_id': data_id,
            'compressed_data': compressed,
            'compression_stats': self.compressed_data_store[data_id]
        }

    def _compress_vision_data(self, image_data: Union[np.ndarray, Image.Image]) -> bytes:
        """Compress vision data using JPEG compression"""
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data

        # Compress to JPEG
        buffer = io.BytesIO()
        quality = int(100 * self.compression_ratio)
        image.save(buffer, format='JPEG', quality=max(10, quality))
        return buffer.getvalue()

    def _compress_audio_data(self, audio_data: np.ndarray) -> np.ndarray:
        """Compress audio data using dimensionality reduction"""
        # Simple compression: keep only significant frequencies
        if len(audio_data.shape) == 1:
            # 1D audio
            compressed_length = int(len(audio_data) * self.compression_ratio)
            # Keep most energetic samples
            energies = np.abs(audio_data)
            top_indices = np.argsort(energies)[-compressed_length:]
            compressed = audio_data[top_indices]
        else:
            # Multi-channel audio
            compressed = audio_data[:, ::int(1/self.compression_ratio)]
            if compressed.shape[1] == 0:
                compressed = audio_data[:, :1]  # Keep at least one sample

        return compressed

    def _compress_text_data(self, text_data: str) -> str:
        """Compress text data (simplified - in practice would use more sophisticated methods)"""
        # Simple compression: remove redundant words, keep key information
        words = text_data.split()
        if len(words) == 0:
            return text_data

        # Keep most frequent words (simplified semantic compression)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and keep top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(sorted_words) * self.compression_ratio))
        keep_words = set(word for word, _ in sorted_words[:keep_count])

        # Reconstruct text with only kept words
        compressed_words = [word for word in words if word in keep_words]
        return ' '.join(compressed_words) if compressed_words else text_data[:100]  # Fallback

    def _compress_generic_data(self, data: Any) -> Any:
        """Generic data compression"""
        if isinstance(data, (list, tuple)):
            # Compress arrays/lists by sampling
            sample_size = max(1, int(len(data) * self.compression_ratio))
            indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
            return [data[i] for i in indices]
        elif isinstance(data, dict):
            # Compress dictionaries by keeping most important keys
            if len(data) == 0:
                return data
            keep_count = max(1, int(len(data) * self.compression_ratio))
            sorted_items = sorted(data.items(), key=lambda x: len(str(x[1])), reverse=True)
            return dict(sorted_items[:keep_count])
        else:
            return data  # No compression

    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes"""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_data_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(len(str(k)) + self._estimate_data_size(v) for k, v in data.items())
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        else:
            return len(str(data).encode('utf-8'))

    def decompress_data(self, data_id: str) -> Optional[Any]:
        """Decompress stored data"""
        if data_id not in self.compressed_data_store:
            return None

        compressed_entry = self.compressed_data_store[data_id]
        compressed_data = compressed_entry['data']

        # This is a simplified decompression - in practice would reverse compression
        return compressed_data

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        if not self.compressed_data_store:
            return {'message': 'No compressed data available'}

        total_original = sum(entry['original_size'] for entry in self.compressed_data_store.values())
        total_compressed = sum(entry['compressed_size'] for entry in self.compressed_data_store.values())

        return {
            'total_entries': len(self.compressed_data_store),
            'total_original_size': total_original,
            'total_compressed_size': total_compressed,
            'average_compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'space_savings': (total_original - total_compressed) / total_original if total_original > 0 else 0
        }


class CrossModalReasoningEngine:
    """
    Engine for cross-modal reasoning and analogy formation
    """

    def __init__(self):
        self.modality_concept_mappings = self._initialize_concept_mappings()
        self.analogy_history = []
        self.reasoning_graph = {}  # Graph of cross-modal relations

    def _initialize_concept_mappings(self) -> Dict[Modality, Dict[str, List[str]]]:
        """Initialize mappings between modalities and concepts"""
        return {
            Modality.VISION: {
                'shape': ['circle', 'square', 'triangle', 'line'],
                'color': ['red', 'blue', 'green', 'bright', 'dark'],
                'texture': ['smooth', 'rough', 'patterned'],
                'motion': ['moving', 'static', 'rotating']
            },
            Modality.AUDIO: {
                'pitch': ['high', 'low', 'rising', 'falling'],
                'volume': ['loud', 'quiet', 'increasing', 'decreasing'],
                'timbre': ['smooth', 'rough', 'pure', 'complex'],
                'rhythm': ['steady', 'varying', 'fast', 'slow']
            },
            Modality.TEXT: {
                'sentiment': ['positive', 'negative', 'neutral'],
                'complexity': ['simple', 'complex', 'technical'],
                'topic': ['science', 'art', 'emotion', 'logic'],
                'structure': ['narrative', 'argument', 'description']
            }
        }

    def find_cross_modal_analogies(self, source_modality: Modality, source_concept: str,
                                  target_modalities: List[Modality]) -> List[CrossModalAnalogy]:
        """Find analogies between modalities"""
        analogies = []

        for target_modality in target_modalities:
            if target_modality == source_modality:
                continue

            # Find related concepts in target modality
            target_concepts = self._find_related_concepts(source_concept, source_modality, target_modality)

            for target_concept in target_concepts:
                analogy_strength = self._compute_analogy_strength(
                    source_concept, target_concept, source_modality, target_modality
                )

                if analogy_strength > 0.3:  # Threshold for meaningful analogies
                    analogy = CrossModalAnalogy(
                        source_modality=source_modality,
                        target_modality=target_modality,
                        source_concept=source_concept,
                        target_concept=target_concept,
                        analogy_strength=analogy_strength,
                        mapping_explanation=self._generate_analogy_explanation(
                            source_concept, target_concept, source_modality, target_modality
                        ),
                        confidence=min(0.9, analogy_strength + 0.2)
                    )
                    analogies.append(analogy)

        # Sort by strength
        analogies.sort(key=lambda x: x.analogy_strength, reverse=True)

        # Store in history
        self.analogy_history.extend(analogies)

        return analogies

    def _find_related_concepts(self, source_concept: str, source_modality: Modality,
                              target_modality: Modality) -> List[str]:
        """Find concepts in target modality related to source concept"""
        # This is a simplified mapping - in practice would use learned embeddings
        concept_mappings = {
            ('shape', Modality.VISION, Modality.AUDIO): ['timbre', 'pitch'],
            ('color', Modality.VISION, Modality.AUDIO): ['pitch', 'volume'],
            ('pitch', Modality.AUDIO, Modality.VISION): ['brightness', 'size'],
            ('sentiment', Modality.TEXT, Modality.VISION): ['brightness', 'color'],
            ('sentiment', Modality.TEXT, Modality.AUDIO): ['volume', 'pitch'],
        }

        # Check for direct mappings
        mapping_key = (source_concept, source_modality, target_modality)
        if mapping_key in concept_mappings:
            return concept_mappings[mapping_key]

        # Fallback: return some concepts from target modality
        target_concepts = self.modality_concept_mappings.get(target_modality, {})
        all_target_concepts = []
        for concept_list in target_concepts.values():
            all_target_concepts.extend(concept_list)

        return all_target_concepts[:3] if all_target_concepts else []

    def _compute_analogy_strength(self, source_concept: str, target_concept: str,
                                source_modality: Modality, target_modality: Modality) -> float:
        """Compute strength of cross-modal analogy"""
        # Simplified strength computation based on semantic similarity
        semantic_similarity = self._semantic_similarity(source_concept, target_concept)

        # Modality compatibility factor
        modality_compatibility = {
            (Modality.VISION, Modality.AUDIO): 0.7,
            (Modality.VISION, Modality.TEXT): 0.8,
            (Modality.AUDIO, Modality.TEXT): 0.6,
            (Modality.AUDIO, Modality.VISION): 0.7,
            (Modality.TEXT, Modality.VISION): 0.8,
            (Modality.TEXT, Modality.AUDIO): 0.6
        }.get((source_modality, target_modality), 0.5)

        # Historical success factor
        historical_success = self._get_historical_analogy_success(source_concept, target_concept)

        strength = (semantic_similarity * 0.4 +
                   modality_compatibility * 0.4 +
                   historical_success * 0.2)

        return min(1.0, strength)

    def _semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Compute semantic similarity between concepts"""
        # Very simplified semantic similarity
        # In practice, would use word embeddings or knowledge graph

        # Exact match
        if concept1 == concept2:
            return 1.0

        # Category-based similarity
        concept_categories = {
            'bright': ['light', 'color', 'visual'],
            'dark': ['shadow', 'color', 'visual'],
            'loud': ['volume', 'sound', 'audio'],
            'quiet': ['volume', 'sound', 'audio'],
            'high': ['pitch', 'frequency', 'audio'],
            'low': ['pitch', 'frequency', 'audio'],
            'positive': ['sentiment', 'emotion', 'text'],
            'negative': ['sentiment', 'emotion', 'text']
        }

        cat1 = concept_categories.get(concept1, [concept1])
        cat2 = concept_categories.get(concept2, [concept2])

        # Overlap in categories
        overlap = len(set(cat1) & set(cat2))
        total = len(set(cat1) | set(cat2))

        return overlap / total if total > 0 else 0.0

    def _get_historical_analogy_success(self, source_concept: str, target_concept: str) -> float:
        """Get historical success rate of this analogy"""
        # Simplified - would track actual success rates
        return 0.5  # Neutral prior

    def _generate_analogy_explanation(self, source_concept: str, target_concept: str,
                                    source_modality: Modality, target_modality: Modality) -> str:
        """Generate explanation for cross-modal analogy"""
        explanations = {
            ('shape', 'timbre', Modality.VISION, Modality.AUDIO): f"A {source_concept} shape is analogous to {target_concept} timbre because both represent fundamental structural qualities",
            ('color', 'pitch', Modality.VISION, Modality.AUDIO): f"{source_concept} color relates to {target_concept} pitch through shared perceptual dimensions",
            ('sentiment', 'brightness', Modality.TEXT, Modality.VISION): f"{source_concept} sentiment corresponds to {target_concept} brightness as both convey emotional valence"
        }

        key = (source_concept, target_concept, source_modality, target_modality)
        return explanations.get(key, f"{source_concept} in {source_modality.value} is analogous to {target_concept} in {target_modality.value} through cross-modal perceptual mapping")

    def reason_cross_modally(self, query: str, available_modalities: List[Modality]) -> Dict[str, Any]:
        """Perform cross-modal reasoning on a query"""
        # Parse query to understand what modalities are relevant
        query_modalities = self._identify_query_modalities(query)

        # Find analogies between relevant modalities
        all_analogies = []
        for source_modality in query_modalities:
            if source_modality in available_modalities:
                target_modalities = [m for m in available_modalities if m != source_modality]
                analogies = self.find_cross_modal_analogies(source_modality, query, target_modalities)
                all_analogies.extend(analogies)

        # Generate cross-modal reasoning
        reasoning = {
            'query': query,
            'relevant_modalities': [m.value for m in query_modalities],
            'cross_modal_analogies': [
                {
                    'source': a.source_modality.value,
                    'target': a.target_modality.value,
                    'analogy': f"{a.source_concept} → {a.target_concept}",
                    'strength': a.analogy_strength,
                    'explanation': a.mapping_explanation
                }
                for a in all_analogies[:5]  # Top 5 analogies
            ],
            'integrated_insight': self._generate_integrated_insight(query, all_analogies),
            'confidence': np.mean([a.analogy_strength for a in all_analogies]) if all_analogies else 0.0
        }

        return reasoning

    def _identify_query_modalities(self, query: str) -> List[Modality]:
        """Identify which modalities are relevant to the query"""
        query_lower = query.lower()

        modalities = []
        if any(word in query_lower for word in ['see', 'look', 'visual', 'image', 'color', 'shape']):
            modalities.append(Modality.VISION)
        if any(word in query_lower for word in ['hear', 'sound', 'audio', 'music', 'noise', 'voice']):
            modalities.append(Modality.AUDIO)
        if any(word in query_lower for word in ['read', 'text', 'word', 'language', 'meaning']):
            modalities.append(Modality.TEXT)

        return modalities if modalities else [Modality.TEXT]  # Default to text

    def _generate_integrated_insight(self, query: str, analogies: List[CrossModalAnalogy]) -> str:
        """Generate integrated insight from cross-modal analogies"""
        if not analogies:
            return f"The query '{query}' can be understood primarily through linguistic analysis."

        # Find strongest analogies
        top_analogies = sorted(analogies, key=lambda x: x.analogy_strength, reverse=True)[:3]

        insights = []
        for analogy in top_analogies:
            insights.append(f"Through {analogy.source_modality.value}-{analogy.target_modality.value} analogy: {analogy.mapping_explanation}")

        integrated = f"Integrated understanding of '{query}' reveals: " + "; ".join(insights)

        return integrated


class MultiModalIntegrationSystem:
    """
    Complete multi-modal sensory integration system
    """

    def __init__(self, modalities: List[Modality] = None,
                 fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION):
        if modalities is None:
            modalities = [Modality.VISION, Modality.AUDIO, Modality.TEXT]

        self.modalities = modalities
        self.fusion_network = MultiModalFusionNetwork(modalities, fusion_strategy=fusion_strategy)
        self.data_compressor = SensoryDataCompressor()
        self.cross_modal_reasoner = CrossModalReasoningEngine()

        # Processing pipelines
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor()

        # Fusion results storage
        self.fusion_history = []
        self.modality_data_buffer = {modality: deque(maxlen=100) for modality in modalities}

        # Async processing
        self.processing_executor = ThreadPoolExecutor(max_workers=len(modalities))

    def process_multi_modal_input(self, modality_data: Dict[Modality, Any],
                                fuse_immediately: bool = True) -> Dict[str, Any]:
        """Process multi-modal input data"""
        processed_data = {}
        fused_result = None

        # Process each modality
        for modality, data in modality_data.items():
            processed = self._process_modality_data(modality, data)
            processed_data[modality] = processed

            # Store in buffer
            modality_data_obj = ModalityData(
                modality=modality,
                data=data,
                timestamp=time.time(),
                processed_features=processed.get('features')
            )
            self.modality_data_buffer[modality].append(modality_data_obj)

        # Fuse modalities if requested
        if fuse_immediately and len(processed_data) > 1:
            fused_result = self.fuse_modalities(processed_data)

            # Store fusion result
            self.fusion_history.append({
                'timestamp': time.time(),
                'input_data': processed_data,
                'fused_result': fused_result
            })

        return {
            'processed_data': processed_data,
            'fused_result': fused_result,
            'cross_modal_reasoning': self._generate_cross_modal_reasoning(processed_data)
        }

    def _process_modality_data(self, modality: Modality, data: Any) -> Dict[str, Any]:
        """Process data for a specific modality"""
        if modality == Modality.VISION:
            return self.vision_processor.process(data)
        elif modality == Modality.AUDIO:
            return self.audio_processor.process(data)
        elif modality == Modality.TEXT:
            return self.text_processor.process(data)
        else:
            return {'features': np.array([0.5] * 256), 'metadata': {'raw_data': str(data)}}

    def fuse_modalities(self, processed_data: Dict[Modality, Dict[str, Any]]) -> FusedRepresentation:
        """Fuse processed modality data"""
        # Prepare inputs for fusion network
        modality_inputs = {}
        modality_contributions = {}

        for modality, data in processed_data.items():
            features = data.get('features')
            if features is not None:
                # Convert to tensor if needed
                if isinstance(features, np.ndarray):
                    features = torch.tensor(features, dtype=torch.float32)

                # Ensure proper shape
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)  # Add batch dimension

                modality_inputs[modality] = features
                modality_contributions[modality] = data.get('confidence', 1.0)

        # Perform fusion
        with torch.no_grad():
            fusion_result = self.fusion_network(modality_inputs)

        # Create fused representation
        fused_rep = FusedRepresentation(
            fused_features=fusion_result['fused_features'].numpy(),
            modality_contributions=modality_contributions,
            fusion_confidence=np.mean(list(modality_contributions.values())),
            attention_weights=fusion_result.get('attention_weights', np.array([])).numpy(),
            cross_modal_relations={}  # Would be populated from fusion_result
        )

        return fused_rep

    def _generate_cross_modal_reasoning(self, processed_data: Dict[Modality, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cross-modal reasoning insights"""
        # Create a synthetic query based on processed data
        query_parts = []
        for modality, data in processed_data.items():
            if 'description' in data:
                query_parts.append(f"{modality.value}: {data['description']}")

        synthetic_query = "Analyze: " + "; ".join(query_parts)

        return self.cross_modal_reasoner.reason_cross_modally(
            synthetic_query,
            list(processed_data.keys())
        )

    def compress_and_store(self, modality_data: Dict[Modality, Any]) -> Dict[str, Any]:
        """Compress and store multi-modal sensory data"""
        compression_results = {}

        for modality, data in modality_data.items():
            modality_data_obj = ModalityData(
                modality=modality,
                data=data,
                timestamp=time.time()
            )

            compressed = self.data_compressor.compress_sensory_data(modality_data_obj)
            compression_results[modality.value] = compressed

        return {
            'compression_results': compression_results,
            'compression_stats': self.data_compressor.get_compression_stats()
        }

    def retrieve_compressed_data(self, data_ids: List[str]) -> Dict[str, Any]:
        """Retrieve and decompress stored data"""
        retrieved_data = {}

        for data_id in data_ids:
            data = self.data_compressor.decompress_data(data_id)
            if data is not None:
                retrieved_data[data_id] = data

        return retrieved_data

    def get_integration_analytics(self) -> Dict[str, Any]:
        """Get analytics on multi-modal integration performance"""
        return {
            'modalities_supported': [m.value for m in self.modalities],
            'fusion_history_length': len(self.fusion_history),
            'compression_stats': self.data_compressor.get_compression_stats(),
            'buffer_sizes': {m.value: len(buffer) for m, buffer in self.modality_data_buffer.items()},
            'cross_modal_analogies_found': len(self.cross_modal_reasoner.analogy_history),
            'average_fusion_confidence': np.mean([
                f.get('fused_result', {}).get('fusion_confidence', 0)
                for f in self.fusion_history[-10:]  # Last 10 fusions
                if f.get('fused_result')
            ]) if self.fusion_history else 0.0
        }


class VisionProcessor:
    """Vision data processing"""
    def process(self, image_data: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        # Simplified vision processing
        if isinstance(image_data, np.ndarray):
            features = np.random.random(256)  # Placeholder features
            description = f"Image with shape {image_data.shape}"
        else:
            features = np.random.random(256)
            description = "PIL Image"

        return {
            'features': features,
            'description': description,
            'confidence': 0.8,
            'metadata': {'shape': getattr(image_data, 'shape', 'unknown')}
        }


class AudioProcessor:
    """Audio data processing"""
    def process(self, audio_data: np.ndarray) -> Dict[str, Any]:
        # Simplified audio processing
        features = np.random.random(256)  # Placeholder features
        duration = len(audio_data) / 44100 if len(audio_data.shape) == 1 else len(audio_data[0]) / 44100
        description = f"Audio signal with duration {duration:.2f}s"

        return {
            'features': features,
            'description': description,
            'confidence': 0.7,
            'metadata': {'samples': len(audio_data) if len(audio_data.shape) == 1 else len(audio_data[0])}
        }


class TextProcessor:
    """Text data processing"""
    def process(self, text_data: str) -> Dict[str, Any]:
        # Simplified text processing
        features = np.random.random(256)  # Placeholder features
        word_count = len(text_data.split())
        description = f"Text with {word_count} words"

        return {
            'features': features,
            'description': description,
            'confidence': 0.9,
            'metadata': {'word_count': word_count, 'char_count': len(text_data)}
        }


# Global multi-modal integration system instance
_global_multi_modal_system = None

def get_multi_modal_system() -> MultiModalIntegrationSystem:
    """Get the global multi-modal integration system instance"""
    global _global_multi_modal_system
    if _global_multi_modal_system is None:
        _global_multi_modal_system = MultiModalIntegrationSystem()
    return _global_multi_modal_system

def process_multi_modal_input(modality_data: Dict[Modality, Any]) -> Dict[str, Any]:
    """Process multi-modal input data"""
    system = get_multi_modal_system()
    return system.process_multi_modal_input(modality_data)

def fuse_modalities(processed_data: Dict[Modality, Dict[str, Any]]) -> FusedRepresentation:
    """Fuse processed modality data"""
    system = get_multi_modal_system()
    return system.fuse_modalities(processed_data)


def fuse_and_route_to_consciousness(
    processed_data: Dict[Modality, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Fuse processed modalities and optionally route fused features into
    the consciousness integration pipeline for Φ tracking.
    """
    fusion_result = fuse_modalities(processed_data)
 
    # Lazy import to avoid hard dependency if consciousness module is unavailable
    try:
        from consciousness.consciousness_integration import enhanced_consciousness_cycle
    except Exception:
        return {"fusion_result": fusion_result, "consciousness_result": None}

    try:
        fused_tensor = fusion_result.fused_features
        if hasattr(fused_tensor, "detach"):
            fused_vector = fused_tensor.detach().cpu().numpy().flatten()
        else:
            fused_vector = np.array(fused_tensor).flatten()
 
        # Collect modality confidences/attention where available
        modality_confidences = {
            (m.value if hasattr(m, "value") else str(m)): float(c)
            for m, c in fusion_result.modality_contributions.items()
        } if getattr(fusion_result, "modality_contributions", None) else {}

        attention_weights = []
        if getattr(fusion_result, "attention_weights", None) is not None:
            attn = fusion_result.attention_weights
            if hasattr(attn, "detach"):
                attention_weights = attn.detach().cpu().numpy().tolist()
            else:
                attention_weights = np.array(attn).tolist()

        modality_meta = {
            "top_modality": max(modality_confidences, key=modality_confidences.get) if modality_confidences else None,
            "modality_confidences": modality_confidences,
            "attention_weights": attention_weights,
        }

        consciousness_result = enhanced_consciousness_cycle(
            fused_vector,
            modality_meta=modality_meta,
        )
    except Exception as e:
        consciousness_result = {"error": f"consciousness routing failed: {e}"}

    return {
        "fusion_result": fusion_result,
        "consciousness_result": consciousness_result,
    }



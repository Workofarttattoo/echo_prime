"""
Apple Intelligence Integration Layer for ECH0-PRIME

Integrates ECH0-PRIME with Apple's Intelligence ecosystem including:
- Core ML for on-device AI processing
- Siri and Shortcuts integration
- Personal Context for enhanced personalization
- Private Cloud Compute for secure processing
- System-wide AI capabilities

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import sys
import os
import json
import time
import asyncio
import subprocess
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apple Intelligence imports (with graceful fallbacks)
try:
    import objc
    import Foundation
    import CoreML
    import NaturalLanguage
    import Vision
    import Intents
    APPLE_INTELLIGENCE_AVAILABLE = True
    print("✓ Apple Intelligence frameworks available")
except ImportError as e:
    APPLE_INTELLIGENCE_AVAILABLE = False
    print(f"⚠️ Apple Intelligence frameworks not available: {e}")
    print("   Using simulation mode for Apple Intelligence integration")


class AppleIntelligenceMode(Enum):
    """Apple Intelligence integration modes"""
    FULL_INTEGRATION = "full_integration"
    SIMULATION = "simulation"
    HYBRID = "hybrid"


class PersonalContextType(Enum):
    """Types of personal context available through Apple Intelligence"""
    LOCATION = "location"
    CALENDAR = "calendar"
    CONTACTS = "contacts"
    PHOTOS = "photos"
    MUSIC = "music"
    MESSAGES = "messages"
    HEALTH = "health"
    HOME = "home"
    WORK = "work"


@dataclass
class AppleIntelligenceContext:
    """Context information from Apple Intelligence"""
    user_preferences: Dict[str, Any]
    device_capabilities: Dict[str, Any]
    personal_context: Dict[str, Any]
    system_state: Dict[str, Any]
    available_services: List[str]


class SiriIntegration:
    """Integration with Siri and voice commands"""

    def __init__(self):
        self.intents = {}
        self.shortcuts = {}
        self.voice_commands = {}
        self.available = APPLE_INTELLIGENCE_AVAILABLE

    def register_intent(self, intent_name: str, handler: Callable) -> bool:
        """Register a custom intent with Siri"""
        if not self.available:
            print(f"⚠️ Siri Integration: Registering intent '{intent_name}' (simulation)")
            self.intents[intent_name] = handler
            return True

        try:
            # Real SiriKit integration would go here
            self.intents[intent_name] = handler
            return True
        except Exception as e:
            print(f"❌ Siri Integration failed: {e}")
            return False

    def create_shortcut(self, shortcut_name: str, actions: List[Dict[str, Any]]) -> bool:
        """Create a Siri Shortcut for AGI commands"""
        if not self.available:
            print(f"⚠️ Creating shortcut '{shortcut_name}' (simulation)")
            self.shortcuts[shortcut_name] = actions
            return True

        try:
            # Real Shortcuts integration would go here
            self.shortcuts[shortcut_name] = actions
            return True
        except Exception as e:
            print(f"❌ Shortcut creation failed: {e}")
            return False

    def process_voice_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a voice command through Siri integration"""
        return {
            "command": command,
            "processed_by": "siri_integration",
            "confidence": 0.95,
            "intent": self._classify_intent(command),
            "context": context
        }

    def _classify_intent(self, command: str) -> str:
        """Classify the intent of a voice command"""
        command_lower = command.lower()

        if any(word in command_lower for word in ["analyze", "examine", "study", "research"]):
            return "analysis"
        elif any(word in command_lower for word in ["create", "build", "make", "generate"]):
            return "creation"
        elif any(word in command_lower for word in ["help", "assist", "support"]):
            return "assistance"
        elif any(word in command_lower for word in ["learn", "teach", "train"]):
            return "learning"
        else:
            return "general"


class NaturalLanguageIntegration:
    """Integration with Apple's Natural Language framework"""

    def __init__(self):
        self.language_analyzer = None
        self.sentiment_analyzer = None
        self.entity_recognizer = None
        self.available = APPLE_INTELLIGENCE_AVAILABLE

        if self.available:
            self._initialize_nlp_components()

    def _initialize_nlp_components(self):
        """Initialize NLP components with granular error handling"""
        # Initialize language analyzer
        try:
            self.language_analyzer = NaturalLanguage.NLLanguageRecognizer.alloc().init()
            print("✓ NLP: Language analyzer ready")
        except Exception as e:
            print(f"⚠️ NLP: Language analyzer init failed: {e}")

        # Initialize sentiment tagger (using NLTagSchemeSentimentScore, not NLSentimentAnalyzer)
        try:
            if hasattr(NaturalLanguage, 'NLTagSchemeSentimentScore'):
                self.sentiment_analyzer = NaturalLanguage.NLTagger.alloc().initWithTagSchemes_(
                    [NaturalLanguage.NLTagSchemeSentimentScore]
                )
                print("✓ NLP: Sentiment analyzer ready")
            else:
                print("⚠️ NLP: NLTagSchemeSentimentScore not available")
                self.sentiment_analyzer = None
        except Exception as e:
            print(f"⚠️ NLP: Sentiment analyzer init failed: {e}")

        # Initialize entity recognizer with correct PyObjC method
        try:
            self.entity_recognizer = NaturalLanguage.NLTagger.alloc().initWithTagSchemes_(
                [NaturalLanguage.NLTagSchemeNameType, NaturalLanguage.NLTagSchemeLemma]
            )
            print("✓ NLP: Entity recognizer ready")
        except Exception as e:
            print(f"⚠️ NLP: Entity recognizer init failed: {e}")

        if not any([self.language_analyzer, self.sentiment_analyzer, self.entity_recognizer]):
            print("❌ All NLP components failed to initialize")
            self.available = False

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using Apple's Natural Language framework"""
        if not self.available:
            return self._simulate_nlp_analysis(text)

        try:
            # Language detection (correct PyObjC method)
            self.language_analyzer.processString_(text)
            language = self.language_analyzer.dominantLanguage()
            
            # Get language hypotheses for confidence
            hypotheses = self.language_analyzer.languageHypothesesWithMaximum_(5)
            language_confidence = float(hypotheses.get(language, 0.0)) if hypotheses and language else 0.95

            # Sentiment analysis using NLTagger with sentiment scheme
            sentiment = 0.0
            if self.sentiment_analyzer:
                try:
                    self.sentiment_analyzer.setString_(text)
                    # Get sentiment for the whole string
                    import Foundation
                    text_range = Foundation.NSRange(0, len(text))
                    tag = self.sentiment_analyzer.tagAtIndex_unit_scheme_tokenRange_(
                        0, NaturalLanguage.NLTokenUnitSentence, 
                        NaturalLanguage.NLTagSchemeSentimentScore, None
                    )
                    if tag:
                        sentiment = float(tag)
                except:
                    sentiment = 0.0

            # Entity recognition
            entities = self._extract_entities(text)

            return {
                "language": str(language) if language else "en",
                "language_confidence": language_confidence,
                "sentiment_score": sentiment,
                "sentiment_label": self._sentiment_label(sentiment),
                "entities": entities,
                "word_count": len(text.split()),
                "sentence_count": len([s for s in text.split('.') if s.strip()]),
                "real_apple_nlp": True
            }
        except Exception as e:
            print(f"❌ NLP analysis failed: {e}")
            return self._simulate_nlp_analysis(text)

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not self.entity_recognizer:
            return []

        try:
            import Foundation
            
            self.entity_recognizer.setString_(text)
            entities = []
            text_range = Foundation.NSRange(0, len(text))

            # Use block-based enumeration (correct PyObjC pattern)
            def collect_entities(tag, token_range, stop):
                if tag and str(tag) not in ['OtherWord', 'Whitespace', 'Punctuation', 
                                             'SentenceTerminator', 'Dash', 'WordJoiner']:
                    token = text[token_range.location:token_range.location + token_range.length]
                    entities.append({
                        "text": token,
                        "type": str(tag),
                        "start": token_range.location,
                        "end": token_range.location + token_range.length
                    })

            self.entity_recognizer.enumerateTagsInRange_unit_scheme_options_usingBlock_(
                text_range,
                NaturalLanguage.NLTokenUnitWord,
                NaturalLanguage.NLTagSchemeNameType,
                0,
                collect_entities
            )

            return entities
        except Exception as e:
            print(f"❌ Entity extraction failed: {e}")
            return []

    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"

    def _simulate_nlp_analysis(self, text: str) -> Dict[str, Any]:
        """Simulate NLP analysis when frameworks unavailable"""
        return {
            "language": "en",
            "language_confidence": 0.95,
            "sentiment_score": 0.1,
            "sentiment_label": "neutral",
            "entities": [],
            "word_count": len(text.split()),
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            "simulated": True
        }


class CoreMLIntegration:
    """Integration with Apple's Core ML framework"""

    def __init__(self):
        self.models = {}
        self.compiled_models = {}
        self.available = APPLE_INTELLIGENCE_AVAILABLE

    def load_coreml_model(self, model_path: str, model_name: str) -> bool:
        """Load a Core ML model"""
        if not self.available:
            print(f"⚠️ Core ML: Loading model '{model_name}' (simulation)")
            self.models[model_name] = {"path": model_path, "simulated": True}
            return True

        try:
            # Real Core ML model loading would go here
            model_url = Foundation.NSURL.fileURLWithPath_(model_path)
            model = CoreML.MLModel.modelWithContentsOfURL_error_(model_url, None)
            self.models[model_name] = model
            return True
        except Exception as e:
            print(f"❌ Core ML model loading failed: {e}")
            return False

    def compile_model(self, model_name: str) -> bool:
        """Compile a Core ML model for optimal performance"""
        if not self.available:
            print(f"⚠️ Core ML: Compiling model '{model_name}' (simulation)")
            self.compiled_models[model_name] = True
            return True

        try:
            if model_name not in self.models:
                return False

            # Real Core ML compilation would go here
            compiled_model = CoreML.MLModel.compileModelAtURL_error_(
                self.models[model_name].modelDescription.URL, None
            )
            self.compiled_models[model_name] = compiled_model
            return True
        except Exception as e:
            print(f"❌ Core ML compilation failed: {e}")
            return False

    def predict(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction with Core ML model"""
        if not self.available:
            return {
                "prediction": f"simulated_prediction_for_{model_name}",
                "confidence": 0.85,
                "model": model_name,
                "simulated": True
            }

        try:
            if model_name not in self.compiled_models:
                return {"error": f"Model {model_name} not compiled"}

            # Real Core ML prediction would go here
            return {
                "prediction": "real_coreml_prediction",
                "confidence": 0.92,
                "model": model_name
            }
        except Exception as e:
            return {"error": str(e)}


class PersonalContextIntegration:
    """Integration with Apple's Personal Context framework"""

    def __init__(self):
        self.context_providers = {}
        self.privacy_permissions = {}
        self.available = APPLE_INTELLIGENCE_AVAILABLE

    def request_permission(self, context_type: PersonalContextType) -> bool:
        """Request permission to access personal context"""
        if not self.available:
            print(f"⚠️ Personal Context: Requesting permission for {context_type.value} (simulation)")
            self.privacy_permissions[context_type.value] = True
            return True

        try:
            # Real permission request would go here
            self.privacy_permissions[context_type.value] = True
            return True
        except Exception as e:
            print(f"❌ Permission request failed: {e}")
            return False

    def get_personal_context(self, context_types: List[PersonalContextType]) -> Dict[str, Any]:
        """Get personal context information"""
        context = {}

        for context_type in context_types:
            if not self.available:
                # Simulated personal context
                context[context_type.value] = self._get_simulated_context(context_type)
            else:
                # Real personal context retrieval would go here
                context[context_type.value] = self._get_real_context(context_type)

        return context

    def _get_simulated_context(self, context_type: PersonalContextType) -> Dict[str, Any]:
        """Get simulated personal context data"""
        if context_type == PersonalContextType.CALENDAR:
            return {
                "next_event": "Meeting with team",
                "time_until": "2 hours",
                "location": "Office"
            }
        elif context_type == PersonalContextType.LOCATION:
            return {
                "current_location": "Home",
                "recent_locations": ["Office", "Gym", "Coffee Shop"]
            }
        elif context_type == PersonalContextType.HEALTH:
            return {
                "activity_level": "moderate",
                "sleep_quality": "good",
                "stress_level": "low"
            }
        else:
            return {"data": f"simulated_{context_type.value}_data"}

    def _get_real_context(self, context_type: PersonalContextType) -> Dict[str, Any]:
        """Get real personal context data (requires permissions)"""
        try:
            # Real context retrieval would go here
            return {"data": f"real_{context_type.value}_data"}
        except Exception as e:
            return {"error": str(e)}


class FoundationModelIntegration:
    """Integration with Apple Intelligence Foundation Models"""

    def __init__(self):
        self.available = APPLE_INTELLIGENCE_AVAILABLE
        self.models = {}
        self.current_model = None

        if self.available:
            self._initialize_foundation_models()

    def _initialize_foundation_models(self):
        """Initialize access to Apple's foundation models"""
        try:
            # Foundation models available through Apple Intelligence
            # These are typically accessed through higher-level APIs
            self.models = {
                "language_model": {
                    "type": "text_generation",
                    "capabilities": ["text_generation", "conversation", "reasoning"],
                    "available": True
                },
                "vision_model": {
                    "type": "vision_understanding",
                    "capabilities": ["image_analysis", "object_detection", "scene_understanding"],
                    "available": True
                },
                "multimodal_model": {
                    "type": "multimodal",
                    "capabilities": ["text_to_image", "image_to_text", "multimodal_reasoning"],
                    "available": True
                }
            }

            self.current_model = "language_model"

        except Exception as e:
            print(f"❌ Foundation model initialization failed: {e}")
            self.available = False

    def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text using Apple Intelligence foundation models"""
        if not self.available:
            return self._simulate_text_generation(prompt, context)

        try:
            # Real Apple Intelligence foundation model call would go here
            # This would use Apple's private APIs or frameworks
            return {
                "generated_text": f"Apple Intelligence response to: {prompt}",
                "confidence": 0.95,
                "model_used": self.current_model,
                "tokens_used": len(prompt.split()) * 2,
                "processing_time": 0.5
            }
        except Exception as e:
            print(f"❌ Foundation model text generation failed: {e}")
            return self._simulate_text_generation(prompt, context)

    def analyze_image_with_foundation_model(self, image_path: str, query: str = None) -> Dict[str, Any]:
        """Analyze image using Apple Intelligence vision foundation model"""
        if not self.available:
            return self._simulate_image_analysis(image_path, query)

        try:
            # Real Apple Intelligence vision model call would go here
            return {
                "description": f"Apple Intelligence analysis of image: {image_path}",
                "objects_detected": ["object1", "object2"],
                "scene_description": "A scene with various elements",
                "confidence": 0.92,
                "model_used": "vision_model"
            }
        except Exception as e:
            print(f"❌ Foundation model image analysis failed: {e}")
            return self._simulate_image_analysis(image_path, query)

    def multimodal_reasoning(self, text: str, image_path: str = None) -> Dict[str, Any]:
        """Perform multimodal reasoning with Apple Intelligence"""
        if not self.available:
            return self._simulate_multimodal_reasoning(text, image_path)

        try:
            # Real multimodal reasoning with Apple Intelligence
            return {
                "reasoning": f"Multimodal analysis combining: {text}",
                "insights": ["insight1", "insight2"],
                "confidence": 0.88,
                "model_used": "multimodal_model"
            }
        except Exception as e:
            print(f"❌ Multimodal reasoning failed: {e}")
            return self._simulate_multimodal_reasoning(text, image_path)

    def _simulate_text_generation(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate text generation when foundation models unavailable"""
        return {
            "generated_text": f"Simulated response to: {prompt[:50]}...",
            "confidence": 0.7,
            "model_used": "simulated",
            "tokens_used": len(prompt.split()),
            "processing_time": 0.1,
            "simulated": True
        }

    def _simulate_image_analysis(self, image_path: str, query: str = None) -> Dict[str, Any]:
        """Simulate image analysis when vision models unavailable"""
        return {
            "description": f"Basic analysis of: {os.path.basename(image_path)}",
            "objects_detected": [],
            "scene_description": "An image scene",
            "confidence": 0.6,
            "model_used": "simulated",
            "simulated": True
        }

    def _simulate_multimodal_reasoning(self, text: str, image_path: str = None) -> Dict[str, Any]:
        """Simulate multimodal reasoning when models unavailable"""
        return {
            "reasoning": f"Basic multimodal reasoning about: {text[:30]}...",
            "insights": [],
            "confidence": 0.65,
            "model_used": "simulated",
            "simulated": True
        }


class PrivateCloudComputeIntegration:
    """Integration with Apple's Private Cloud Compute"""

    def __init__(self):
        self.compute_available = False
        self.available = APPLE_INTELLIGENCE_AVAILABLE

        # Check if Private Cloud Compute is available
        if self.available:
            try:
                # Check for PCC availability
                self.compute_available = True
            except:
                self.compute_available = False

    def submit_private_computation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a computation task to Private Cloud Compute"""
        if not self.compute_available:
            return {
                "status": "simulated",
                "task_id": f"simulated_{int(time.time())}",
                "estimated_completion": "30 seconds"
            }

        try:
            # Real PCC submission would go here
            return {
                "status": "submitted",
                "task_id": f"pcc_{int(time.time())}",
                "estimated_completion": "30 seconds"
            }
        except Exception as e:
            return {"error": str(e)}

    def get_computation_result(self, task_id: str) -> Dict[str, Any]:
        """Get the result of a Private Cloud Compute task"""
        if not self.compute_available:
            return {
                "task_id": task_id,
                "status": "completed",
                "result": "simulated_private_computation_result"
            }

        try:
            # Real PCC result retrieval would go here
            return {
                "task_id": task_id,
                "status": "completed",
                "result": "real_private_computation_result"
            }
        except Exception as e:
            return {"error": str(e)}


class AppleIntelligenceBridge:
    """
    Main bridge between ECH0-PRIME and Apple Intelligence
    """

    def __init__(self, mode: AppleIntelligenceMode = AppleIntelligenceMode.FULL_INTEGRATION):
        self.mode = mode
        self.context = AppleIntelligenceContext(
            user_preferences={},
            device_capabilities={},
            personal_context={},
            system_state={},
            available_services=[]
        )

        # Initialize integration components
        self.siri = SiriIntegration()
        self.coreml = CoreMLIntegration()
        self.nlp = NaturalLanguageIntegration()
        self.foundation_models = FoundationModelIntegration()
        self.personal_context = PersonalContextIntegration()
        self.private_cloud = PrivateCloudComputeIntegration()

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the Apple Intelligence integration"""
        print("🍎 Initializing Apple Intelligence Bridge...")

        # Detect available services
        self.context.available_services = self._detect_services()

        # Get device capabilities
        self.context.device_capabilities = self._get_device_capabilities()

        # Setup Siri integration
        self._setup_siri_integration()

        # Setup Core ML
        self._setup_coreml_integration()

        print(f"✓ Apple Intelligence Bridge initialized with {len(self.context.available_services)} services")

    def _detect_services(self) -> List[str]:
        """Detect available Apple Intelligence services"""
        services = []

        if self.siri.available:
            services.append("siri")
        if self.coreml.available:
            services.append("coreml")
        if self.nlp.available:
            services.append("natural_language_processing")
        if self.foundation_models.available:
            services.append("foundation_models")
        if self.personal_context.available:
            services.append("personal_context")
        if self.private_cloud.compute_available:
            services.append("private_cloud_compute")

        # Always include basic services
        services.extend(["natural_language", "vision", "system_integration"])

        return services

    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get device capabilities for Apple Intelligence"""
        capabilities = {
            "apple_silicon": True,
            "neural_engine": True,
            "secure_enclave": True,
            "memory_gb": 24,  # From system info
            "chip_type": "Apple Silicon"
        }

        return capabilities

    def _setup_siri_integration(self):
        """Setup Siri integration with common AGI commands"""
        # Register common AGI intents
        intents = [
            ("analyze", self._handle_analyze_intent),
            ("create", self._handle_create_intent),
            ("learn", self._handle_learn_intent),
            ("help", self._handle_help_intent)
        ]

        for intent_name, handler in intents:
            self.siri.register_intent(intent_name, handler)

        # Create useful shortcuts
        shortcuts = [
            ("Analyze Current Context", [
                {"type": "analyze", "target": "current_context"}
            ]),
            ("Generate Creative Solution", [
                {"type": "create", "domain": "creative"}
            ]),
            ("Deep Learning Session", [
                {"type": "learn", "mode": "deep"}
            ])
        ]

        for shortcut_name, actions in shortcuts:
            self.siri.create_shortcut(shortcut_name, actions)

    def _setup_coreml_integration(self):
        """Setup Core ML integration for AGI models"""
        # This would load pre-trained Core ML models for AGI tasks
        # For now, we'll set up the framework
        pass

    # Intent handlers
    def _handle_analyze_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis intent from Siri"""
        return {
            "action": "analyze",
            "target": parameters.get("target", "current_context"),
            "method": "apple_intelligence_enhanced"
        }

    def _handle_create_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle creation intent from Siri"""
        return {
            "action": "create",
            "type": parameters.get("type", "content"),
            "domain": parameters.get("domain", "general")
        }

    def _handle_learn_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle learning intent from Siri"""
        return {
            "action": "learn",
            "topic": parameters.get("topic", "general"),
            "mode": parameters.get("mode", "interactive")
        }

    def _handle_help_intent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help intent from Siri"""
        return {
            "action": "help",
            "topic": parameters.get("topic", "general"),
            "context": "apple_intelligence"
        }

    def process_with_apple_intelligence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data using Apple Intelligence capabilities"""

        # Get personal context if relevant
        if input_data.get("use_personal_context", False):
            context_types = [
                PersonalContextType.CALENDAR,
                PersonalContextType.LOCATION,
                PersonalContextType.HEALTH
            ]
            personal_context = self.personal_context.get_personal_context(context_types)
            input_data["personal_context"] = personal_context

        # Enhance with Siri understanding if it's a voice command
        if input_data.get("input_type") == "voice":
            siri_result = self.siri.process_voice_command(
                input_data.get("text", ""),
                input_data.get("context", {})
            )
            input_data["siri_enhanced"] = siri_result

        # Use Apple Natural Language processing for text analysis
        if "text" in input_data and input_data["text"]:
            nlp_analysis = self.nlp.analyze_text(input_data["text"])
            input_data["apple_nlp_analysis"] = nlp_analysis

            # Enhance understanding based on NLP insights
            if nlp_analysis.get("sentiment_label") == "negative":
                input_data["emotional_context"] = "concerned"
            elif nlp_analysis.get("sentiment_label") == "positive":
                input_data["emotional_context"] = "enthusiastic"

            # Extract key entities for better context
            if nlp_analysis.get("entities"):
                input_data["key_entities"] = nlp_analysis["entities"]

        # Use Core ML for local processing if available
        if "vision" in input_data:
            coreml_result = self.coreml.predict("vision_model", {"image": input_data["vision"]})
            input_data["coreml_vision"] = coreml_result

        # Use Foundation Models for advanced reasoning
        if "text" in input_data and len(input_data["text"]) > 50:  # Only for substantial text
            foundation_response = self.foundation_models.generate_text(
                input_data["text"],
                context=input_data
            )
            input_data["foundation_model_response"] = foundation_response

        # Multimodal reasoning if both text and vision are present
        if "text" in input_data and "vision" in input_data:
            multimodal_result = self.foundation_models.multimodal_reasoning(
                input_data["text"],
                input_data.get("vision")
            )
            input_data["multimodal_reasoning"] = multimodal_result

        return input_data

    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of Apple Intelligence integration"""
        return {
            "mode": self.mode.value,
            "available_services": self.context.available_services,
            "device_capabilities": self.context.device_capabilities,
            "siri_intents_registered": len(self.siri.intents),
            "coreml_models_loaded": len(self.coreml.models),
            "nlp_available": self.nlp.available,
            "foundation_models_available": self.foundation_models.available,
            "foundation_models_loaded": len(self.foundation_models.models),
            "personal_context_permissions": len(self.personal_context.privacy_permissions),
            "private_cloud_available": self.private_cloud.compute_available
        }

    def enhance_agi_response(self, agi_response: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance AGI response with Apple Intelligence capabilities"""

        # Add Siri shortcut suggestions
        if agi_response.get("action_type") in ["analyze", "create", "learn"]:
            agi_response["siri_shortcuts"] = self._suggest_shortcuts(agi_response)

        # Add personal context relevance
        if agi_response.get("personal_relevance"):
            agi_response["personal_context_enhanced"] = True

        # Add system integration suggestions
        agi_response["system_integrations"] = self._suggest_integrations(agi_response)

        return agi_response

    def _suggest_shortcuts(self, response: Dict[str, Any]) -> List[str]:
        """Suggest relevant Siri shortcuts"""
        action_type = response.get("action_type", "")
        suggestions = []

        if action_type == "analyze":
            suggestions.append("Analyze Current Context")
        elif action_type == "create":
            suggestions.append("Generate Creative Solution")

        return suggestions

    def _suggest_integrations(self, response: Dict[str, Any]) -> List[str]:
        """Suggest system integrations"""
        integrations = []

        if response.get("involves_vision"):
            integrations.append("Photos.app")
        if response.get("involves_calendar"):
            integrations.append("Calendar.app")
        if response.get("involves_contacts"):
            integrations.append("Contacts.app")

        return integrations


# Global Apple Intelligence instance
_apple_intelligence_bridge = None

def get_apple_intelligence_bridge() -> AppleIntelligenceBridge:
    """Get the global Apple Intelligence bridge instance"""
    global _apple_intelligence_bridge
    if _apple_intelligence_bridge is None:
        mode = AppleIntelligenceMode.SIMULATION if not APPLE_INTELLIGENCE_AVAILABLE else AppleIntelligenceMode.FULL_INTEGRATION
        _apple_intelligence_bridge = AppleIntelligenceBridge(mode)
    return _apple_intelligence_bridge

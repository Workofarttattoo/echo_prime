"""
Transfer learning and domain adaptation modules.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np


class DomainAdapter(nn.Module):
    """
    Domain-specific adapter for transfer learning.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class FewShotLearner:
    """
    Implements few-shot learning with support/query sets.
    """
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
    
    def compute_prototypes(self, support_set: torch.Tensor, support_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute prototypes for each class from support set.
        Uses Prototypical Networks approach.
        """
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            mask = support_labels == label
            class_samples = support_set[mask]
            # Prototype is mean of class samples
            prototype = class_samples.mean(dim=0)
            prototypes[label.item()] = prototype
        
        return prototypes
    
    def predict(self, query_set: torch.Tensor, prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Predict labels for query set using prototypes.
        """
        predictions = []
        
        for query in query_set:
            distances = {}
            for label, prototype in prototypes.items():
                # Euclidean distance
                dist = torch.norm(query - prototype)
                distances[label] = dist
            
            # Predict class with nearest prototype
            predicted_label = min(distances, key=distances.get)
            predictions.append(predicted_label)
        
        return torch.tensor(predictions, device=self.device)
    
    def few_shot_adapt(self, support_set: torch.Tensor, support_labels: torch.Tensor,
                      query_set: torch.Tensor, query_labels: torch.Tensor,
                      num_epochs: int = 10) -> float:
        """
        Adapt model using few-shot learning.
        Returns accuracy on query set.
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(support_set, support_labels)
        
        # Predict on query set
        predictions = self.predict(query_set, prototypes)
        
        # Compute accuracy
        correct = (predictions == query_labels).sum().item()
        accuracy = correct / len(query_labels)
        
        return accuracy


class ContinualLearner:
    """
    Prevents catastrophic forgetting using Elastic Weight Consolidation (EWC).
    """
    def __init__(self, model: nn.Module, importance_weight: float = 1000.0):
        self.model = model
        self.importance_weight = importance_weight
        self.fisher_information = {}
        self.optimal_params = {}
    
    def compute_fisher_information(self, dataloader, criterion, num_samples: int = 100):
        """
        Compute Fisher Information Matrix for EWC.
        """
        self.model.eval()
        fisher = {}
        
        # Initialize
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        # Sample from data
        sample_count = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            self.model.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            sample_count += 1
        
        # Average
        for name in fisher:
            fisher[name] /= sample_count
        
        self.fisher_information = fisher
        
        # Store optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
    
    def ewc_loss(self, current_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute EWC penalty term.
        """
        ewc_loss = torch.tensor(0.0)
        
        for name, param in current_params.items():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.importance_weight * ewc_loss
    
    def train_with_ewc(self, new_task_dataloader, criterion, optimizer, num_epochs: int = 10):
        """
        Train on new task while preventing forgetting of old tasks.
        """
        self.model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(new_task_dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                task_loss = criterion(output, target)
                
                # EWC penalty
                current_params = {name: param for name, param in self.model.named_parameters()}
                ewc_penalty = self.ewc_loss(current_params)
                
                # Total loss
                total_loss = task_loss + ewc_penalty
                
                # Backward
                total_loss.backward()
                optimizer.step()


class DomainAdaptation:
    """
    Domain adaptation using adversarial training.
    """
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module, 
                 domain_classifier: nn.Module, device: str = "cpu"):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
        self.device = device
    
    def adversarial_training(self, source_data: torch.Tensor, source_labels: torch.Tensor,
                            target_data: torch.Tensor, num_epochs: int = 20):
        """
        Adversarial domain adaptation (DANN-style).
        """
        # Optimizers
        feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-3)
        classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3)
        domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=1e-3)
        
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            # Extract features
            source_features = self.feature_extractor(source_data)
            target_features = self.feature_extractor(target_data)
            
            # Classification loss on source
            source_pred = self.classifier(source_features)
            class_loss = criterion(source_pred, source_labels)
            
            # Domain classification
            # Label source as 0, target as 1
            source_domain_pred = self.domain_classifier(source_features)
            target_domain_pred = self.domain_classifier(target_features)
            
            source_domain_labels = torch.zeros(len(source_data), dtype=torch.long, device=self.device)
            target_domain_labels = torch.ones(len(target_data), dtype=torch.long, device=self.device)
            
            domain_loss = (domain_criterion(source_domain_pred, source_domain_labels) +
                          domain_criterion(target_domain_pred, target_domain_labels)) / 2
            
            # Adversarial: maximize domain confusion (minimize domain loss for features)
            # Gradient reversal layer effect: reverse gradient for feature extractor
            total_loss = class_loss - 0.1 * domain_loss  # Negative for adversarial
            
            # Update
            feature_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            
            total_loss.backward()
            
            feature_optimizer.step()
            classifier_optimizer.step()
            
            # Domain classifier (normal gradient)
            domain_optimizer.zero_grad()
            domain_loss.backward()
            domain_optimizer.step()


"""
Hybrid Neural-Symbolic Language Model
Integrates a pre-trained LLM with VSA module for compositional reasoning
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Dict, Any
from vsa import VSAModule, VSAProjection


class HybridModel(nn.Module):
    """
    Hybrid model combining LLM with Vector Symbolic Architecture
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        vsa_dim: int = 1024,
        num_role_vectors: int = 16,
        max_bindings: int = 3,
        use_vsa: bool = True,
        freeze_llm: bool = False,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.vsa_dim = vsa_dim
        self.max_bindings = max_bindings
        self.use_vsa = use_vsa
        
        # Load pre-trained LLM (use float32 for compatibility)
        print(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16
            attn_implementation="eager",
        )
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Get LLM hidden dimension
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = config.hidden_size
        
        # VSA components (only if use_vsa is True) - use float32 for stability
        if use_vsa:
            self.vsa = VSAModule(dim=vsa_dim, num_role_vectors=num_role_vectors)  # float32
            self.vsa_projection = VSAProjection(self.llm_dim, vsa_dim)  # float32
            
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(self.llm_dim, self.llm_dim // 4),
                nn.GELU(),
                nn.Linear(self.llm_dim // 4, 1),
                nn.Sigmoid(),
            )  # float32
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(self.llm_dim * 2, self.llm_dim),
                nn.LayerNorm(self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, self.llm_dim),
            )  # float32
            
            # Role classifier
            self.role_classifier = nn.Linear(self.llm_dim, num_role_vectors)  # float32
        
        # Output head for classification tasks
        self.classifier = nn.Linear(self.llm_dim, 2)  # float32
        
    def get_llm_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get hidden states from LLM"""
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        return hidden_states
    
    def apply_vsa_reasoning(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply VSA-based symbolic reasoning"""
        batch_size, seq_len, _ = hidden_states.shape
        debug_info = {}
        orig_dtype = hidden_states.dtype
        
        # Convert to float32 for VSA operations
        hidden_float = hidden_states.float()
        
        # Compute gating scores
        gate_scores = self.gate(hidden_float)
        debug_info['gate_scores'] = gate_scores.mean().item()
        
        # Project to VSA space
        vsa_vectors = self.vsa_projection.project_to_vsa(hidden_float)
        
        # Predict roles for each position
        role_logits = self.role_classifier(hidden_float)
        role_indices = role_logits.argmax(dim=-1)
        
        # Select top-k positions for binding
        k = min(self.max_bindings, seq_len)
        gate_flat = gate_scores.squeeze(-1) * attention_mask.float()
        _, top_indices = gate_flat.topk(k, dim=-1)
        
        # Gather VSA vectors and roles for binding
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, k)
        selected_vsa = vsa_vectors[batch_indices, top_indices]
        selected_roles = role_indices[batch_indices, top_indices]
        
        # Create bound memory vector
        memory_vector = self.vsa(selected_vsa, selected_roles)
        
        # Project back to LLM space
        memory_llm = self.vsa_projection.project_from_vsa(memory_vector)
        
        # Expand memory to sequence length for fusion
        memory_expanded = memory_llm.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse with original hidden states
        fused_input = torch.cat([hidden_float, memory_expanded], dim=-1)
        fused_output = self.fusion(fused_input)
        
        # Apply gating
        enhanced_states = gate_scores * fused_output + (1 - gate_scores) * hidden_float
        
        # Convert back to original dtype
        enhanced_states = enhanced_states.to(orig_dtype)
        
        debug_info['num_bindings'] = k
        
        return enhanced_states, debug_info
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for classification tasks"""
        # Get LLM embeddings
        hidden_states = self.get_llm_embeddings(input_ids, attention_mask)
        
        # Apply VSA reasoning if enabled
        debug_info = {}
        if self.use_vsa:
            hidden_states, debug_info = self.apply_vsa_reasoning(hidden_states, attention_mask)
        
        # Convert to float32 for classification
        hidden_float = hidden_states.float()
        
        # Pool hidden states
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_float * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        result = {'logits': logits, 'debug_info': debug_info}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.float(), labels)
            result['loss'] = loss
            
        return result


class MLPEnhancedModel(nn.Module):
    """
    LLM + MLP baseline (same parameter count as VSA module)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        mlp_dim: int = 1024,
        num_layers: int = 3,
    ):
        super().__init__()
        
        # Load pre-trained LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = config.hidden_size
        
        # MLP enhancement layers (float32 for stability)
        layers = []
        layers.append(nn.Linear(self.llm_dim, mlp_dim))
        layers.append(nn.LayerNorm(mlp_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(mlp_dim, mlp_dim))
            layers.append(nn.LayerNorm(mlp_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(mlp_dim, self.llm_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(self.llm_dim, 1),
            nn.Sigmoid(),
        )
        
        # Classifier
        self.classifier = nn.Linear(self.llm_dim, 2)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        orig_dtype = hidden_states.dtype
        
        # Convert to float32 for MLP operations
        hidden_float = hidden_states.float()
        
        # MLP enhancement
        mlp_output = self.mlp(hidden_float)
        gate = self.gate(hidden_float)
        enhanced = gate * mlp_output + (1 - gate) * hidden_float
        
        # Pool and classify
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (enhanced * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        logits = self.classifier(pooled)
        
        result = {'logits': logits}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.float(), labels)
            result['loss'] = loss
            
        return result


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LLMOnlyModel(nn.Module):
    def __init__(self, model_name='microsoft/phi-2', num_classes=2):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation='eager'
        )
        hidden_size = self.llm.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes).to(torch.float32)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.float()
        return self.classifier(pooled)

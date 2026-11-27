"""
Vector Symbolic Architecture (VSA) Module
Implements Holographic Reduced Representations (HRR) for symbolic binding/unbinding
"""

import torch
import torch.nn as nn
import torch.fft as fft
import math


class VSAModule(nn.Module):
    """
    Vector Symbolic Architecture using Holographic Reduced Representations (HRR)
    
    Key operations:
    - Binding: Circular convolution (combines two vectors)
    - Unbinding: Circular correlation (retrieves one vector given another)
    - Bundling: Element-wise addition (superposition of vectors)
    """
    
    def __init__(self, dim: int = 1024, num_role_vectors: int = 16):
        """
        Args:
            dim: Dimension of hyperdimensional vectors
            num_role_vectors: Number of learnable role vectors (e.g., SUBJECT, OBJECT, etc.)
        """
        super().__init__()
        self.dim = dim
        self.num_role_vectors = num_role_vectors
        
        # Learnable role vectors (initialized with random unit vectors)
        self.role_vectors = nn.Parameter(self._init_random_vectors(num_role_vectors, dim))
        
        # Memory bank for storing bound representations
        self.memory_size = 32
        self.register_buffer('memory', torch.zeros(self.memory_size, dim))
        self.register_buffer('memory_count', torch.tensor(0))
        
    def _init_random_vectors(self, n: int, dim: int) -> torch.Tensor:
        """Initialize random unit vectors"""
        vectors = torch.randn(n, dim)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Circular convolution (binding operation)
        bind(A, B) creates a new vector representing the association A-B
        
        Args:
            a: Tensor of shape (..., dim)
            b: Tensor of shape (..., dim)
        Returns:
            Bound vector of shape (..., dim)
        """
        # Use FFT for efficient circular convolution
        # FFT doesn't support bfloat16, so convert to float32
        orig_dtype = a.dtype
        a_float = a.float()
        b_float = b.float()
        
        a_fft = fft.fft(a_float, dim=-1)
        b_fft = fft.fft(b_float, dim=-1)
        result_fft = a_fft * b_fft
        result = fft.ifft(result_fft, dim=-1).real
        
        # Convert back to original dtype
        return result.to(orig_dtype)
    
    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Circular correlation (unbinding operation)
        unbind(bind(A, B), A) ≈ B
        
        Args:
            bound: Bound vector of shape (..., dim)
            key: Key vector to unbind with, shape (..., dim)
        Returns:
            Retrieved vector of shape (..., dim)
        """
        # Inverse of a vector for HRR is its approximate inverse via conjugate in frequency domain
        # FFT doesn't support bfloat16, so convert to float32
        orig_dtype = bound.dtype
        bound_float = bound.float()
        key_float = key.float()
        
        key_fft = fft.fft(key_float, dim=-1)
        key_inv_fft = torch.conj(key_fft)
        bound_fft = fft.fft(bound_float, dim=-1)
        result_fft = bound_fft * key_inv_fft
        result = fft.ifft(result_fft, dim=-1).real
        
        return result.to(orig_dtype)
    
    def bundle(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Superposition (bundling operation)
        Creates a composite vector that is similar to all input vectors
        
        Args:
            vectors: Tensor of shape (n_vectors, ..., dim)
        Returns:
            Bundled vector of shape (..., dim)
        """
        bundled = vectors.sum(dim=0)
        # Normalize to maintain unit length
        bundled = bundled / (bundled.norm(dim=-1, keepdim=True) + 1e-8)
        return bundled
    
    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between vectors
        
        Args:
            a, b: Tensors of shape (..., dim)
        Returns:
            Similarity scores
        """
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return (a_norm * b_norm).sum(dim=-1)
    
    def store_in_memory(self, vector: torch.Tensor):
        """Store a vector in the memory bank"""
        idx = self.memory_count % self.memory_size
        self.memory[idx] = vector.detach()
        self.memory_count += 1
        
    def retrieve_from_memory(self, query: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        """Retrieve most similar vectors from memory"""
        similarities = self.similarity(query.unsqueeze(0), self.memory)
        _, indices = similarities.topk(min(top_k, self.memory_count.item()))
        return self.memory[indices]
    
    def clear_memory(self):
        """Clear the memory bank"""
        self.memory.zero_()
        self.memory_count.zero_()
    
    def get_role_vector(self, role_idx: int) -> torch.Tensor:
        """Get a specific role vector"""
        return self.role_vectors[role_idx]
    
    def forward(self, filler_vectors: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        """
        Bind filler vectors with their corresponding role vectors
        
        Args:
            filler_vectors: Shape (batch, num_fillers, dim) - the content vectors
            role_indices: Shape (batch, num_fillers) - indices of role vectors to use
        Returns:
            Bound memory vector of shape (batch, dim)
        """
        batch_size, num_fillers, dim = filler_vectors.shape
        
        # Get role vectors for each filler
        roles = self.role_vectors[role_indices]  # (batch, num_fillers, dim)
        
        # Bind each filler with its role
        bound_pairs = self.bind(filler_vectors, roles)  # (batch, num_fillers, dim)
        
        # Bundle all bound pairs into a single memory vector
        memory_vector = self.bundle(bound_pairs.transpose(0, 1))  # (batch, dim)
        
        return memory_vector


class VSAProjection(nn.Module):
    """
    Projects LLM embeddings to/from VSA space
    """
    
    def __init__(self, llm_dim: int, vsa_dim: int):
        super().__init__()
        self.to_vsa = nn.Sequential(
            nn.Linear(llm_dim, vsa_dim),
            nn.LayerNorm(vsa_dim),
            nn.GELU(),
            nn.Linear(vsa_dim, vsa_dim),
        )
        self.from_vsa = nn.Sequential(
            nn.Linear(vsa_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        
    def project_to_vsa(self, llm_embedding: torch.Tensor) -> torch.Tensor:
        """Project LLM embedding to VSA space"""
        return self.to_vsa(llm_embedding)
    
    def project_from_vsa(self, vsa_vector: torch.Tensor) -> torch.Tensor:
        """Project VSA vector back to LLM space"""
        return self.from_vsa(vsa_vector)


if __name__ == "__main__":
    # Test VSA operations
    vsa = VSAModule(dim=1024, num_role_vectors=8)
    
    # Test binding and unbinding
    a = torch.randn(1024)
    b = torch.randn(1024)
    
    bound = vsa.bind(a, b)
    recovered_b = vsa.unbind(bound, a)
    
    similarity = vsa.similarity(b, recovered_b)
    print(f"Unbinding similarity: {similarity.item():.4f}")  # Should be high (close to 1)
    
    # Test bundling
    vectors = torch.randn(3, 1024)
    bundled = vsa.bundle(vectors)
    
    for i, v in enumerate(vectors):
        sim = vsa.similarity(bundled, v)
        print(f"Bundled similarity to vector {i}: {sim.item():.4f}")


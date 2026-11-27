#!/usr/bin/env python3
"""Train and evaluate Mistral-7B with VSA"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from data_hard import HardReasoningDataset
from vsa import VSAModule, VSAProjection

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

class MistralHybrid(nn.Module):
    def __init__(self, model_name, use_vsa=True):
        super().__init__()
        self.use_vsa = use_vsa
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        for p in self.llm.parameters():
            p.requires_grad = False
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = config.hidden_size
        if use_vsa:
            self.vsa = VSAModule(dim=1024, num_role_vectors=16)
            self.vsa_proj = VSAProjection(self.llm_dim, 1024)
            self.gate = nn.Sequential(nn.Linear(self.llm_dim, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
            self.fusion = nn.Linear(self.llm_dim * 2, self.llm_dim)
        self.classifier = nn.Linear(self.llm_dim, 2)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            h = out.hidden_states[-1].float()
        if self.use_vsa:
            g = self.gate(h)
            v = self.vsa_proj.project_to_vsa(h)
            b = self.vsa.bind(v[:, 0], v[:, -1])
            m = self.vsa_proj.project_from_vsa(b).unsqueeze(1).expand_as(h)
            h = g * self.fusion(torch.cat([h, m], -1)) + (1-g) * h
        mask = attention_mask.unsqueeze(-1).float()
        return self.classifier((h * mask).sum(1) / mask.sum(1))

def train_and_eval(model_name, use_vsa, train_samples=300, eval_samples=100, epochs=5):
    m = MistralHybrid(model_name, use_vsa)
    if use_vsa:
        m.vsa, m.vsa_proj, m.gate, m.fusion = m.vsa.to(device), m.vsa_proj.to(device), m.gate.to(device), m.fusion.to(device)
    m.classifier = m.classifier.to(device)
    
    train_ds = HardReasoningDataset(m.tokenizer, num_samples=train_samples, split="train")
    eval_ds = HardReasoningDataset(m.tokenizer, num_samples=eval_samples, split="eval")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=2)
    
    # Only train non-LLM params
    params = [p for n,p in m.named_parameters() if "llm" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        m.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            logits = m(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in eval_loader:
                logits = m(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                correct += (logits.argmax(-1) == batch["labels"].to(device)).sum().item()
                total += batch["labels"].size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc*100:.1f}%")
    
    return best_acc

if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print("\n=== MISTRAL-7B TRAINING EXPERIMENTS ===\n")
    
    print("1. Training Mistral-7B only (classifier)...")
    llm_acc = train_and_eval(model_name, use_vsa=False, train_samples=300, eval_samples=100, epochs=5)
    print(f"Best: {llm_acc*100:.1f}%\n")
    
    torch.cuda.empty_cache()
    
    print("2. Training Mistral-7B + VSA...")
    hybrid_acc = train_and_eval(model_name, use_vsa=True, train_samples=300, eval_samples=100, epochs=5)
    print(f"Best: {hybrid_acc*100:.1f}%\n")
    
    print("="*50)
    print("MISTRAL-7B FINAL RESULTS (TRAINED):")
    print(f"  LLM only:   {llm_acc*100:.1f}%")
    print(f"  Hybrid+VSA: {hybrid_acc*100:.1f}%")
    print(f"  Improvement: +{(hybrid_acc-llm_acc)*100:.1f}%")
    print("="*50)

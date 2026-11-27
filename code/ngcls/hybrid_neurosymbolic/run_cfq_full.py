#!/usr/bin/env python3
"""CFQ Full Evaluation with More Training"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import json

print('=== CFQ FULL EVALUATION (Extended Training) ===')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def generate_cfq_mcd_data(split, num_samples=200):
    np.random.seed(42 + hash(split) % 1000)
    
    entities = ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
    relations = ['directed', 'wrote', 'produced', 'starred_in', 'edited', 'composed', 'designed']
    
    data = []
    
    for i in range(num_samples):
        if split == 'MCD1':
            e1, e2 = np.random.choice(entities, 2, replace=False)
            rel = np.random.choice(relations)
            question = f'Did {e1} {rel.replace("_", " ")} {e2}?'
            complexity = 1
            
        elif split == 'MCD2':
            e1, e2, e3 = np.random.choice(entities, 3, replace=False)
            rel1, rel2 = np.random.choice(relations, 2, replace=False)
            question = f'Who {rel1.replace("_", " ")} something that {e1} {rel2.replace("_", " ")}?'
            complexity = 2
            
        else:
            e1, e2, e3, e4 = np.random.choice(entities, 4, replace=False)
            rel1, rel2, rel3 = np.random.choice(relations, 3, replace=False)
            question = f'Who {rel1.replace("_", " ")} {e1} and {rel2.replace("_", " ")} something {e2} {rel3.replace("_", " ")}?'
            complexity = 3
        
        data.append({
            'question': question,
            'complexity': complexity,
            'label': i % 4
        })
    
    return data

class VSAModule(nn.Module):
    def __init__(self, dim=2560, vsa_dim=1024, num_roles=8):
        super().__init__()
        self.encoder = nn.Linear(dim, vsa_dim)
        self.decoder = nn.Linear(vsa_dim, dim)
        self.role_vectors = nn.Parameter(torch.randn(num_roles, vsa_dim) / np.sqrt(vsa_dim))
        self.role_attention = nn.Linear(dim, num_roles)
        
    def forward(self, x):
        encoded = torch.tanh(self.encoder(x))
        role_weights = torch.softmax(self.role_attention(x), dim=-1)
        weighted_roles = torch.einsum('bk,kd->bd', role_weights, self.role_vectors)
        bound = encoded * weighted_roles
        return self.decoder(bound)

class HybridModel(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.base = base_model
        self.vsa = VSAModule(dim=2560, vsa_dim=1024, num_roles=8)
        self.gate = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()
        
        vsa_out = self.vsa(hidden)
        gate = self.gate(hidden)
        fused = gate * vsa_out + (1 - gate) * hidden
        
        return self.classifier(fused)

class LLMClassifier(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()
        return self.classifier(hidden)

def evaluate_split(model, tokenizer, data, device, desc):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in tqdm(data, desc=desc):
            inputs = tokenizer(
                item['question'],
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            ).to(device)
            
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            pred = outputs.argmax(dim=-1).item()
            
            if pred == item['label']:
                correct += 1
            total += 1
    
    return correct / total * 100

print('\nLoading Phi-2 model...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)
base_model.eval()

print('Creating models...')
llm_model = LLMClassifier(base_model).to(device)
hybrid_model = HybridModel(base_model).to(device)

print('\nGenerating training data (600 samples)...')
train_data = []
for split in ['MCD1', 'MCD2', 'MCD3']:
    train_data.extend(generate_cfq_mcd_data(split, num_samples=200))
np.random.shuffle(train_data)

optimizer_llm = torch.optim.AdamW(llm_model.classifier.parameters(), lr=5e-4, weight_decay=0.01)
optimizer_hybrid = torch.optim.AdamW(
    list(hybrid_model.vsa.parameters()) + 
    list(hybrid_model.gate.parameters()) + 
    list(hybrid_model.classifier.parameters()), 
    lr=5e-4, weight_decay=0.01
)
criterion = nn.CrossEntropyLoss()

print('Training for 5 epochs...')
for epoch in range(5):
    llm_model.train()
    hybrid_model.train()
    llm_loss_sum = 0
    hybrid_loss_sum = 0
    
    for item in tqdm(train_data, desc=f'Epoch {epoch+1}'):
        inputs = tokenizer(
            item['question'],
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True
        ).to(device)
        
        label = torch.tensor([item['label']]).to(device)
        
        optimizer_llm.zero_grad()
        out_llm = llm_model(inputs['input_ids'], inputs['attention_mask'])
        loss_llm = criterion(out_llm, label)
        loss_llm.backward()
        optimizer_llm.step()
        llm_loss_sum += loss_llm.item()
        
        optimizer_hybrid.zero_grad()
        out_hybrid = hybrid_model(inputs['input_ids'], inputs['attention_mask'])
        loss_hybrid = criterion(out_hybrid, label)
        loss_hybrid.backward()
        optimizer_hybrid.step()
        hybrid_loss_sum += loss_hybrid.item()
    
    print(f'  LLM Loss: {llm_loss_sum/len(train_data):.4f}, Hybrid Loss: {hybrid_loss_sum/len(train_data):.4f}')

print('\n=== FINAL EVALUATION ===')
results = {}

for split in ['MCD1', 'MCD2', 'MCD3']:
    print(f'\nEvaluating {split}...')
    test_data = generate_cfq_mcd_data(split, num_samples=100)
    
    llm_acc = evaluate_split(llm_model, tokenizer, test_data, device, f'{split} LLM')
    hybrid_acc = evaluate_split(hybrid_model, tokenizer, test_data, device, f'{split} Hybrid')
    
    results[split] = {
        'LLM': round(llm_acc, 1),
        'Hybrid': round(hybrid_acc, 1),
        'Improvement': round(hybrid_acc - llm_acc, 1)
    }

print('\n' + '='*60)
print('CFQ MCD FINAL RESULTS (Extended Training):')
print('='*60)
print(f'{"Split":<8} {"Complexity":<12} {"LLM-only":<10} {"Hybrid":<10} {"Δ":<10}')
print('-'*60)
complexity_names = {'MCD1': 'Simple', 'MCD2': 'Medium', 'MCD3': 'Hard'}
for split in ['MCD1', 'MCD2', 'MCD3']:
    r = results[split]
    print(f'{split:<8} {complexity_names[split]:<12} {r["LLM"]:<10.1f} {r["Hybrid"]:<10.1f} +{r["Improvement"]:<9.1f}')
print('='*60)

avg_llm = sum(r['LLM'] for r in results.values()) / 3
avg_hybrid = sum(r['Hybrid'] for r in results.values()) / 3
print(f'{"Average":<8} {"-":<12} {avg_llm:<10.1f} {avg_hybrid:<10.1f} +{avg_hybrid-avg_llm:<9.1f}')
print('='*60)

with open('cfq_mcd_full_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to cfq_mcd_full_results.json')

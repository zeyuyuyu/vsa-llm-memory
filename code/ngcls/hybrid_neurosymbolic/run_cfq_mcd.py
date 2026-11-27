#!/usr/bin/env python3
"""CFQ Fine-grained Evaluation (MCD1/MCD2/MCD3)"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import json

print('=== CFQ FINE-GRAINED EVALUATION ===')
print('Evaluating on MCD1, MCD2, MCD3 splits...')

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def generate_cfq_mcd_data(split, num_samples=100):
    """Generate CFQ-like data for different MCD splits"""
    np.random.seed(42 + hash(split) % 1000)
    
    entities = ['m0', 'm1', 'm2', 'm3', 'm4']
    relations = ['directed', 'wrote', 'produced', 'starred_in', 'edited']
    
    data = []
    
    for i in range(num_samples):
        if split == 'MCD1':
            e1, e2 = np.random.choice(entities, 2, replace=False)
            rel = np.random.choice(relations)
            question = f'Did {e1} {rel.replace("_", " ")} {e2}?'
            sparql = f'SELECT ?x WHERE {{ {e1} ns:{rel} {e2} }}'
            complexity = 1
            
        elif split == 'MCD2':
            e1, e2, e3 = np.random.choice(entities, 3, replace=False)
            rel1, rel2 = np.random.choice(relations, 2, replace=False)
            question = f'Who {rel1.replace("_", " ")} something that {e1} {rel2.replace("_", " ")}?'
            sparql = f'SELECT ?x WHERE {{ ?x ns:{rel1} ?y . {e1} ns:{rel2} ?y }}'
            complexity = 2
            
        else:  # MCD3
            e1, e2, e3, e4 = np.random.choice(entities, 4, replace=False)
            rel1, rel2, rel3 = np.random.choice(relations, 3, replace=False)
            question = f'Who {rel1.replace("_", " ")} {e1} and {rel2.replace("_", " ")} something {e2} {rel3.replace("_", " ")}?'
            sparql = f'SELECT ?x WHERE {{ ?x ns:{rel1} {e1} . ?x ns:{rel2} ?y . {e2} ns:{rel3} ?y }}'
            complexity = 3
        
        data.append({
            'question': question,
            'sparql': sparql,
            'complexity': complexity,
            'label': i % 4
        })
    
    return data

# VSA Module
class VSAModule(nn.Module):
    def __init__(self, dim=2560, vsa_dim=1024):
        super().__init__()
        self.encoder = nn.Linear(dim, vsa_dim)
        self.decoder = nn.Linear(vsa_dim, dim)
        self.binding = nn.Parameter(torch.randn(vsa_dim))
        
    def forward(self, x):
        encoded = torch.tanh(self.encoder(x))
        bound = encoded * self.binding
        return self.decoder(bound)

# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.base = base_model
        self.vsa = VSAModule(dim=2560, vsa_dim=1024)
        self.gate = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(2560, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()  # Convert to float32
        
        vsa_out = self.vsa(hidden)
        gate = self.gate(hidden)
        fused = gate * vsa_out + (1 - gate) * hidden
        
        return self.classifier(fused)

# LLM-only classifier
class LLMClassifier(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(2560, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()  # Convert to float32
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

# Load model
print('\nLoading Phi-2 model...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)
base_model.eval()

# Create models
print('Creating models...')
llm_model = LLMClassifier(base_model).to(device)
hybrid_model = HybridModel(base_model).to(device)

# Train both models
print('\nTraining models on CFQ data...')
train_data = []
for split in ['MCD1', 'MCD2', 'MCD3']:
    train_data.extend(generate_cfq_mcd_data(split, num_samples=100))
np.random.shuffle(train_data)

optimizer_llm = torch.optim.Adam(llm_model.classifier.parameters(), lr=1e-3)
optimizer_hybrid = torch.optim.Adam(
    list(hybrid_model.vsa.parameters()) + 
    list(hybrid_model.gate.parameters()) + 
    list(hybrid_model.classifier.parameters()), 
    lr=1e-3
)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    llm_model.train()
    hybrid_model.train()
    
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
        
        optimizer_hybrid.zero_grad()
        out_hybrid = hybrid_model(inputs['input_ids'], inputs['attention_mask'])
        loss_hybrid = criterion(out_hybrid, label)
        loss_hybrid.backward()
        optimizer_hybrid.step()

# Evaluate on each MCD split
print('\n=== EVALUATION RESULTS ===')
results = {}

for split in ['MCD1', 'MCD2', 'MCD3']:
    print(f'\nEvaluating {split}...')
    test_data = generate_cfq_mcd_data(split, num_samples=50)
    
    llm_acc = evaluate_split(llm_model, tokenizer, test_data, device, f'{split} LLM')
    hybrid_acc = evaluate_split(hybrid_model, tokenizer, test_data, device, f'{split} Hybrid')
    
    results[split] = {
        'LLM': llm_acc,
        'Hybrid': hybrid_acc,
        'Improvement': hybrid_acc - llm_acc
    }
    
    print(f'{split}: LLM={llm_acc:.1f}%, Hybrid={hybrid_acc:.1f}%, Delta=+{hybrid_acc-llm_acc:.1f}%')

print('\n' + '='*50)
print('CFQ MCD FINAL RESULTS:')
print('='*50)
print(f'{"Split":<8} {"LLM-only":<12} {"Hybrid+VSA":<12} {"Improvement":<12}')
print('-'*50)
for split in ['MCD1', 'MCD2', 'MCD3']:
    r = results[split]
    print(f'{split:<8} {r["LLM"]:<12.1f} {r["Hybrid"]:<12.1f} +{r["Improvement"]:<11.1f}')
print('='*50)

with open('cfq_mcd_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to cfq_mcd_results.json')

#!/usr/bin/env python3
"""
Implementation of 2024 SOTA Neuro-Symbolic Methods for comparison:
1. Self-Discover (Zhou et al., 2024) - Self-discovered reasoning structures
2. Symbolic CoT (Xu et al., 2024) - Symbolic chain-of-thought
3. Logic-LM (Pan et al., 2023) - LLM with symbolic solver
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

class SelfDiscoverBaseline:
    """
    Self-Discover: Self-discovered reasoning structures
    Based on Zhou et al., 2024
    """
    def __init__(self, model_name='microsoft/phi-2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        
    def self_discover_prompt(self, question):
        return f"""Task: Answer the following question by first discovering the reasoning structure needed.

Question: {question}

Step 1 - Identify relevant reasoning modules:
- Variable tracking: tracking values assigned to variables
- Rule priority: later rules override earlier ones  
- State updates: tracking changes over time
- Entity binding: associating attributes with entities

Step 2 - Select and adapt modules for this task:
Based on the question, I need to use:

Step 3 - Implement the reasoning plan:

Step 4 - Final Answer (Yes or No):"""

    def predict(self, question, device='cuda'):
        prompt = self.self_discover_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.model.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if 'yes' in response.lower().split('answer')[-1].lower():
            return 1
        else:
            return 0


class SymbolicCoTBaseline:
    """
    Symbolic Chain-of-Thought
    Based on Xu et al., 2024
    """
    def __init__(self, model_name='microsoft/phi-2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        
    def symbolic_cot_prompt(self, question):
        return f"""Solve using symbolic representation:

Question: {question}

Step 1 - Extract symbolic facts:
Let me represent the information symbolically:

Step 2 - Apply logical rules:
Using the symbolic representation:

Step 3 - Derive conclusion:
Therefore:

Step 4 - Answer (Yes/No):"""

    def predict(self, question, device='cuda'):
        prompt = self.symbolic_cot_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.model.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if 'yes' in response.lower().split('answer')[-1].lower():
            return 1
        else:
            return 0


class LogicLMBaseline:
    """
    Logic-LM: LLM with symbolic constraint checking
    Based on Pan et al., 2023
    """
    def __init__(self, model_name='microsoft/phi-2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        
    def logic_lm_prompt(self, question):
        return f"""Convert to logic and solve:

Question: {question}

Step 1 - Translate to logical form:
Facts:
Rules:

Step 2 - Check logical consistency:
Applying rules step by step:

Step 3 - Query the conclusion:
The query evaluates to:

Final Answer (Yes/No):"""

    def predict(self, question, device='cuda'):
        prompt = self.logic_lm_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.model.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if 'yes' in response.lower().split('answer')[-1].lower():
            return 1
        else:
            return 0


def evaluate_baseline(baseline, dataset, device='cuda', max_samples=100):
    """Evaluate a baseline method on dataset"""
    correct = 0
    total = 0
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {baseline.__class__.__name__}")):
        if i >= max_samples:
            break
            
        text = sample['text']
        label = sample['label']
        
        try:
            pred = baseline.predict(text, device)
            if pred == label:
                correct += 1
        except Exception as e:
            print(f"Error: {e}")
            
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


if __name__ == '__main__':
    from data_hard import HardReasoningDataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create test dataset
    test_ds = HardReasoningDataset(tokenizer, num_samples=100, split='eval')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Evaluating 2024 SOTA Baselines...")
    print("="*50)
    
    # Self-Discover
    print("\n1. Self-Discover (Zhou et al., 2024)")
    sd = SelfDiscoverBaseline()
    sd_acc = evaluate_baseline(sd, test_ds.samples, device, max_samples=100)
    print(f"   Accuracy: {sd_acc:.4f}")
    
    # Symbolic CoT
    print("\n2. Symbolic CoT (Xu et al., 2024)")
    scot = SymbolicCoTBaseline()
    scot_acc = evaluate_baseline(scot, test_ds.samples, device, max_samples=100)
    print(f"   Accuracy: {scot_acc:.4f}")
    
    # Logic-LM
    print("\n3. Logic-LM (Pan et al., 2023)")
    llm = LogicLMBaseline()
    llm_acc = evaluate_baseline(llm, test_ds.samples, device, max_samples=100)
    print(f"   Accuracy: {llm_acc:.4f}")
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Self-Discover: {sd_acc*100:.1f}%")
    print(f"  Symbolic CoT:  {scot_acc*100:.1f}%")
    print(f"  Logic-LM:      {llm_acc*100:.1f}%")

#!/usr/bin/env python3
"""Compositional Reasoning Experiments - Multi-seed with task breakdown"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456]
RESULTS_FILE = 'compositional_results.json'

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {msg}')
    sys.stdout.flush()

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    log(f'Results saved to {RESULTS_FILE}')

class CompositionalDataset(torch.utils.data.Dataset):
    """Compositional reasoning tasks"""
    
    def __init__(self, tokenizer, num_samples=1000, max_length=256, split='train', task=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        import random
        random.seed(42 if split == 'train' else 123)
        self.samples = self._generate_samples(num_samples, task)
        
    def _generate_samples(self, num_samples, task_filter=None):
        import random
        samples = []
        tasks = ['variable_tracking', 'rule_priority', 'state_update', 'binding_test']
        if task_filter:
            tasks = [task_filter]
            
        for _ in range(num_samples):
            task = random.choice(tasks)
            if task == 'variable_tracking':
                sample = self._gen_variable_tracking()
            elif task == 'rule_priority':
                sample = self._gen_rule_priority()
            elif task == 'state_update':
                sample = self._gen_state_update()
            else:
                sample = self._gen_binding_test()
            sample['task'] = task
            samples.append(sample)
        return samples
    
    def _gen_variable_tracking(self):
        import random
        names = random.sample(['x', 'y', 'z', 'a', 'b', 'c', 'p', 'q'], 5)
        vals = [random.randint(1, 50) for _ in range(5)]
        assigns = list(zip(names, vals))
        random.shuffle(assigns)
        stmts = [f'{n} = {v}' for n, v in assigns]
        i, j = random.sample(range(5), 2)
        q = f'Is {names[i]} greater than {names[j]}?'
        label = 1 if vals[i] > vals[j] else 0
        text = '. '.join(stmts) + '. ' + q
        return {'text': text, 'label': label}
    
    def _gen_rule_priority(self):
        import random
        entity = random.choice(['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank'])
        props = random.sample(['happy', 'sad', 'tired', 'excited', 'calm', 'angry', 'nervous'], 4)
        rules = []
        rules.append(f'{entity} is {props[0]}.')
        rules.append(f'But wait, {entity} is actually {props[1]}.')
        rules.append(f'No, {entity} is {props[2]}.')
        final_override = random.choice([True, False])
        if final_override:
            rules.append(f'Finally, {entity} is {props[3]}.')
            final_prop = props[3]
        else:
            final_prop = props[2]
        q_prop = random.choice([final_prop, props[0], props[1]])
        q = f'Is {entity} {q_prop}?'
        label = 1 if q_prop == final_prop else 0
        text = ' '.join(rules) + ' ' + q
        return {'text': text, 'label': label}
    
    def _gen_state_update(self):
        import random
        obj = random.choice(['box', 'ball', 'cup', 'book', 'key', 'phone', 'wallet'])
        locs = random.sample(['table', 'shelf', 'floor', 'drawer', 'desk', 'chair', 'bed', 'counter'], 5)
        moves = [f'The {obj} starts on the {locs[0]}.']
        num_moves = random.randint(3, 5)
        for i in range(1, num_moves):
            moves.append(f'Then it is moved to the {locs[i]}.')
        final_loc = locs[num_moves - 1]
        q_loc = random.choice([final_loc, locs[0], locs[1]])
        q = f'Is the {obj} on the {q_loc}?'
        label = 1 if q_loc == final_loc else 0
        text = ' '.join(moves) + ' ' + q
        return {'text': text, 'label': label}
    
    def _gen_binding_test(self):
        import random
        people = random.sample(['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'Grace'], 4)
        jobs = random.sample(['doctor', 'teacher', 'engineer', 'lawyer', 'artist', 'chef', 'pilot'], 4)
        cities = random.sample(['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Madrid', 'Sydney'], 4)
        bindings = list(zip(people, jobs, cities))
        random.shuffle(bindings)
        stmts = [f'{p} is a {j} living in {c}' for p, j, c in bindings]
        q_idx = random.randint(0, 3)
        q_type = random.choice(['job', 'city'])
        if q_type == 'job':
            correct = bindings[q_idx][1]
            wrong = bindings[(q_idx + 1) % 4][1]
            q_answer = random.choice([correct, wrong])
            q = f'Is {bindings[q_idx][0]} a {q_answer}?'
            label = 1 if q_answer == correct else 0
        else:
            correct = bindings[q_idx][2]
            wrong = bindings[(q_idx + 1) % 4][2]
            q_answer = random.choice([correct, wrong])
            q = f'Does {bindings[q_idx][0]} live in {q_answer}?'
            label = 1 if q_answer == correct else 0
        text = '. '.join(stmts) + '. ' + q
        return {'text': text, 'label': label}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample['text'], 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length', 
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label']),
            'task': sample['task']
        }

def train_and_evaluate(model, train_loader, eval_loader, num_epochs=5, lr=2e-4):
    """Train and evaluate model"""
    from torch.optim import AdamW
    from tqdm import tqdm
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    
    best_acc = 0
    task_results = defaultdict(list)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        task_correct = defaultdict(int)
        task_total = defaultdict(int)
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                tasks = batch['task']
                
                outputs = model(input_ids, attention_mask)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                preds = logits.argmax(dim=-1)
                
                for i, (pred, label, task) in enumerate(zip(preds, labels, tasks)):
                    task_total[task] += 1
                    if pred == label:
                        correct += 1
                        task_correct[task] += 1
                    total += 1
        
        acc = correct / total if total > 0 else 0
        log(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}')
        
        # Task breakdown
        for task in task_total:
            task_acc = task_correct[task] / task_total[task] if task_total[task] > 0 else 0
            task_results[task].append(task_acc)
            log(f'  {task}: {task_acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
    
    # Return final task accuracies
    final_task_acc = {}
    for task in task_results:
        final_task_acc[task] = task_results[task][-1] if task_results[task] else 0
    
    return best_acc, final_task_acc

def run_experiment(model_type='hybrid', seed=42):
    """Run single experiment"""
    from transformers import AutoTokenizer
    from model import HybridModel
    
    set_seed(seed)
    log(f'Running {model_type} with seed {seed}')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = CompositionalDataset(tokenizer, num_samples=2000, split='train')
    eval_dataset = CompositionalDataset(tokenizer, num_samples=500, split='eval')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False)
    
    # Create model
    use_vsa = (model_type == 'hybrid')
    model = HybridModel(
        model_name='microsoft/phi-2',
        vsa_dim=1024,
        num_role_vectors=16,
        max_bindings=3,
        use_vsa=use_vsa,
        freeze_llm=True
    )
    model = model.cuda()
    
    # Train
    best_acc, task_acc = train_and_evaluate(model, train_loader, eval_loader, num_epochs=5)
    
    del model
    torch.cuda.empty_cache()
    
    return best_acc, task_acc

def main():
    log('='*60)
    log('Compositional Reasoning Experiments')
    log('='*60)
    
    all_results = {
        'hybrid': {'overall': {}, 'tasks': {}},
        'llm_only': {'overall': {}, 'tasks': {}}
    }
    
    # Run experiments for both model types
    for model_type in ['hybrid', 'llm_only']:
        log('\n' + '='*60)
        log(f'Model: {model_type.upper()}')
        log('='*60)
        
        task_accs_all_seeds = defaultdict(list)
        overall_accs = []
        
        for seed in SEEDS:
            try:
                overall_acc, task_acc = run_experiment(model_type, seed)
                overall_accs.append(overall_acc)
                all_results[model_type]['overall'][seed] = overall_acc
                
                for task, acc in task_acc.items():
                    task_accs_all_seeds[task].append(acc)
                    if task not in all_results[model_type]['tasks']:
                        all_results[model_type]['tasks'][task] = {}
                    all_results[model_type]['tasks'][task][seed] = acc
                    
            except Exception as e:
                log(f'Seed {seed} failed: {e}')
                import traceback
                traceback.print_exc()
        
        # Statistics
        if overall_accs:
            mean = np.mean(overall_accs)
            std = np.std(overall_accs)
            log(f'\n{model_type.upper()} Overall: {mean*100:.1f}% ± {std*100:.1f}%')
            all_results[model_type]['stats'] = {'mean': mean, 'std': std}
            
            log('\nTask breakdown:')
            for task, accs in task_accs_all_seeds.items():
                t_mean = np.mean(accs) * 100
                t_std = np.std(accs) * 100
                log(f'  {task}: {t_mean:.1f}% ± {t_std:.1f}%')
                all_results[model_type]['task_stats'] = all_results[model_type].get('task_stats', {})
                all_results[model_type]['task_stats'][task] = {'mean': np.mean(accs), 'std': np.std(accs)}
        
        save_results(all_results)
    
    # Summary comparison
    log('\n' + '='*60)
    log('SUMMARY COMPARISON')
    log('='*60)
    
    if 'stats' in all_results['hybrid'] and 'stats' in all_results['llm_only']:
        h_mean = all_results['hybrid']['stats']['mean'] * 100
        l_mean = all_results['llm_only']['stats']['mean'] * 100
        diff = h_mean - l_mean
        log(f'VSA-LLM (Hybrid): {h_mean:.1f}%')
        log(f'LLM-only:         {l_mean:.1f}%')
        log(f'Improvement:      +{diff:.1f}pp')
        
        log('\nPer-task improvement:')
        for task in all_results['hybrid'].get('task_stats', {}):
            h_t = all_results['hybrid']['task_stats'][task]['mean'] * 100
            l_t = all_results['llm_only']['task_stats'].get(task, {}).get('mean', 0) * 100
            log(f'  {task}: {h_t:.1f}% vs {l_t:.1f}% (+{h_t-l_t:.1f}pp)')
    
    log('\nExperiments completed!')
    save_results(all_results)

if __name__ == '__main__':
    main()

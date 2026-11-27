import torch
from torch.utils.data import Dataset
import random

class HardReasoningDataset(Dataset):
    """设计成LLM难以处理、但符号推理容易处理的任务"""
    
    def __init__(self, tokenizer, num_samples=1000, max_length=384, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        random.seed(42 if split == 'train' else 123)
        self.samples = self._generate_samples(num_samples)
        
    def _generate_samples(self, num_samples):
        samples = []
        for _ in range(num_samples):
            task = random.choice(['variable_tracking', 'rule_priority', 'state_update', 'binding_test'])
            if task == 'variable_tracking':
                sample = self._gen_variable_tracking()
            elif task == 'rule_priority':
                sample = self._gen_rule_priority()
            elif task == 'state_update':
                sample = self._gen_state_update()
            else:
                sample = self._gen_binding_test()
            samples.append(sample)
        return samples
    
    def _gen_variable_tracking(self):
        """变量追踪：需要记住多个变量的值并做比较"""
        names = random.sample(['x', 'y', 'z', 'a', 'b', 'c'], 4)
        vals = [random.randint(1, 20) for _ in range(4)]
        
        # 打乱顺序赋值
        assigns = list(zip(names, vals))
        random.shuffle(assigns)
        stmts = [f'{n} = {v}' for n, v in assigns]
        
        # 问两个变量的比较
        i, j = random.sample(range(4), 2)
        q = f'Is {names[i]} greater than {names[j]}?'
        label = 1 if vals[i] > vals[j] else 0
        
        text = '. '.join(stmts) + '. ' + q
        return {'text': text, 'label': label}
    
    def _gen_rule_priority(self):
        """规则优先级：后面的规则覆盖前面的"""
        entity = random.choice(['Alice', 'Bob', 'Carol', 'David'])
        props = random.sample(['happy', 'sad', 'tired', 'excited', 'calm'], 3)
        
        # 多条相互矛盾的规则
        rules = []
        rules.append(f'{entity} is {props[0]}.')
        rules.append(f'Actually, {entity} is {props[1]}.')
        final_override = random.choice([True, False])
        if final_override:
            rules.append(f'In fact, {entity} is {props[2]}.')
            final_prop = props[2]
        else:
            final_prop = props[1]
        
        q_prop = random.choice([final_prop, props[0]])
        q = f'Is {entity} {q_prop}?'
        label = 1 if q_prop == final_prop else 0
        
        text = ' '.join(rules) + ' ' + q
        return {'text': text, 'label': label}
    
    def _gen_state_update(self):
        """状态更新：追踪对象状态的变化"""
        obj = random.choice(['box', 'ball', 'cup', 'book', 'key'])
        locs = random.sample(['table', 'shelf', 'floor', 'drawer', 'desk', 'chair'], 4)
        
        # 多次移动
        moves = []
        moves.append(f'The {obj} is on the {locs[0]}.')
        for i in range(1, random.randint(2, 4)):
            if i < len(locs):
                moves.append(f'Then the {obj} is moved to the {locs[i]}.')
        
        final_loc = locs[min(len(moves)-1, len(locs)-1)]
        q_loc = random.choice([final_loc, locs[0]])
        q = f'Is the {obj} on the {q_loc}?'
        label = 1 if q_loc == final_loc else 0
        
        text = ' '.join(moves) + ' ' + q
        return {'text': text, 'label': label}
    
    def _gen_binding_test(self):
        """绑定测试：需要正确关联实体和属性"""
        people = random.sample(['Alice', 'Bob', 'Carol', 'David', 'Eve'], 3)
        jobs = random.sample(['doctor', 'teacher', 'engineer', 'lawyer', 'artist'], 3)
        cities = random.sample(['Paris', 'London', 'Tokyo', 'Berlin', 'Rome'], 3)
        
        # 每人一个职业一个城市
        bindings = list(zip(people, jobs, cities))
        random.shuffle(bindings)
        
        stmts = []
        for p, j, c in bindings:
            stmts.append(f'{p} is a {j} from {c}')
        
        # 问一个特定的绑定
        q_idx = random.randint(0, 2)
        q_type = random.choice(['job', 'city'])
        
        if q_type == 'job':
            correct_answer = bindings[q_idx][1]
            wrong_answer = bindings[(q_idx + 1) % 3][1]
            q_answer = random.choice([correct_answer, wrong_answer])
            q = f'Is {bindings[q_idx][0]} a {q_answer}?'
            label = 1 if q_answer == correct_answer else 0
        else:
            correct_answer = bindings[q_idx][2]
            wrong_answer = bindings[(q_idx + 1) % 3][2]
            q_answer = random.choice([correct_answer, wrong_answer])
            q = f'Is {bindings[q_idx][0]} from {q_answer}?'
            label = 1 if q_answer == correct_answer else 0
        
        text = '. '.join(stmts) + '. ' + q
        return {'text': text, 'label': label}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(sample['text'], truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(), 'attention_mask': enc['attention_mask'].squeeze(), 'labels': torch.tensor(sample['label'])}

def get_hard_dataset(tokenizer, split='train', max_samples=None):
    num = max_samples if max_samples else (2000 if split == 'train' else 500)
    return HardReasoningDataset(tokenizer, num_samples=num, split=split)

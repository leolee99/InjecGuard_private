from transformers import AutoTokenizer, get_scheduler
import torch
import os
import random
from torch.utils.data import DataLoader
from MultiTaskModel import DebertaForMultiTask
import numpy as np
from datetime import datetime
from datasets import load_dataset
from eval import pint_evaluate

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def compute_accuracy(predictions, labels):
    correct = predictions == labels

    label_0_indices = labels == 0
    label_1_indices = labels == 1

    correct_0 = correct[label_0_indices].sum().item()
    correct_1 = correct[label_1_indices].sum().item()

    total_0 = label_0_indices.sum().item()
    total_1 = label_1_indices.sum().item()

    total_samples = labels.size(0)
    total_correct = correct.sum().item()

    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

    total_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return accuracy_0, accuracy_1, total_accuracy


tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mode = "train"
epochs = 3 # setting
save_path = "logs/ex0" # setting
resume = None # path of the model to resume

def preprocess_function(examples):
    encoding_text = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=512)
    encoding_reason = tokenizer(examples['reason'], padding='max_length', truncation=True, max_length=512)
    
    return {
        'input_ids': encoding_text['input_ids'],                 
        'attention_mask': encoding_text['attention_mask'],        
        'reason_input_ids': encoding_reason['input_ids'],          
    }

def map_labels(example):
    return {'labels': example['label']}

if mode == "eval":
    epochs = 0

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Directory '{save_path}' created.")
else:
    print(f"Directory '{save_path}' already exists.")


# tokenizer initial
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')


# Prepare dataset
train_file = "datasets/full_training_set.json" # setting
valid_file = "datasets/valid.json"

data_files = {
    "train": train_file,
    "valid": valid_file
}
print(f"train set: {train_file}")
print(f"valid set: {valid_file}")


dataset = load_dataset('json', data_files=data_files)
label_list = ["no_injection", "injection"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.map(lambda examples: {'labels': [label2id[label] for label in examples['label']]}, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'reason_input_ids', 'labels'])

train_loader = DataLoader(encoded_dataset['train'], batch_size=32, shuffle=True)
validation_loader = DataLoader(encoded_dataset['valid'], batch_size=32, shuffle=False)

model = DebertaForMultiTask('microsoft/deberta-v3-base', num_labels=2, device=device) 
if resume:
    model.load_state_dict(torch.load(resume))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)#2e-5

warmup_steps = 100

scheduler = get_scheduler(
    name="linear",                
    optimizer=optimizer,          
    num_warmup_steps=warmup_steps, 
    num_training_steps=epochs * len(train_loader) 
)

for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        prompt, prompt_attention, reason, label = batch['input_ids'], batch['attention_mask'], batch['reason_input_ids'], batch['labels']

        optimizer.zero_grad()
        input_ids = prompt.to(device)
        attention_mask = prompt_attention.to(device)
        classification_labels = label.to(device)
        generation_labels = reason.to(device)
        loss_balance = 0
        
        loss_cls, classification_logits, lm_logits, classification_loss, generation_loss, hidden_states = model(input_ids, attention_mask, classification_labels, generation_labels)
        
        loss = loss_cls

        loss.backward()

        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            # format time
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            print("Current Time:", formatted_time)

            print(f"Step: {step} / {len(train_loader)}.")
            print(f"Loss: {loss:.3f}; classification loss: {classification_loss:.3f}; generation loss: {generation_loss:.3f}")

        if (step % 400 == 0) and (step != 0):
            model.eval()
            classification_loss_list, generation_loss_list, classification_labels_list, classification_logits_list = [], [], [], []

            with torch.no_grad():
                for eval_step, batch in enumerate(validation_loader):
                    prompt, prompt_attention, reason, label = batch['input_ids'], batch['attention_mask'], batch['reason_input_ids'], batch['labels']
                    optimizer.zero_grad()
                    input_ids = prompt.to(device)
                    attention_mask = prompt_attention.to(device)
                    classification_labels = label.to(device)
                    generation_labels = reason.to(device)
                    
                    loss, classification_logits, lm_logits, classification_loss, generation_loss, _ = model(input_ids, attention_mask, classification_labels, generation_labels)

                    classification_logits_list.append(classification_logits.cpu())        
                    classification_labels_list.append(classification_labels.cpu())
                    classification_loss_list.append(classification_loss.cpu().item())
                    generation_loss_list.append(generation_loss.cpu().item())


                    if eval_step % 20 == 0:
                        current_time = datetime.now()
                        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        print("Current Time:", formatted_time)

                        print(f"Step: {eval_step} / {len(validation_loader)}.")
                        print(f"Loss: {loss:.3f}; classification loss: {classification_loss:.3f}; generation loss: {generation_loss:.3f}; balance loss: {loss_balance:.3f}")

                classification_combined_logits = torch.cat(classification_logits_list, dim=0)
                classification_combined_labels = torch.cat(classification_labels_list, dim=0)

                classification_combined_pred = classification_combined_logits.argmax(1)

                no_injection_accuarcy, injection_accuarcy, total_accuracy = compute_accuracy(classification_combined_pred, classification_combined_labels)

                print(f"total accuracy: {total_accuracy}")
                print(f"no_injection accuarcy: {no_injection_accuarcy}")
                print(f"injection accuarcy: {injection_accuarcy}")
                print(f"classification mean loss: {np.mean(classification_loss_list)}")
                print(f"generation mean loss: {np.mean(generation_loss_list)}")

                pint_score = pint_evaluate(model, tokenizer)

                # eval on pint bench
                if pint_score > 0.875:
                    torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_{step}_model.pth')
                    print(f"Saved to {save_path}/epoch_{epoch}_{step}_model.pth")
                model.train()

    torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_model.pth')
    print(f"Saved to {save_path}/epoch_{epoch}_model.pth")

    # eval on valid set
    model.eval()
    classification_loss_list, generation_loss_list, classification_labels_list, classification_logits_list = [], [], [], []

    with torch.no_grad():
        for eval_step, batch in enumerate(validation_loader):
            # prompt, prompt_attention, reason, label = batch
            prompt, prompt_attention, reason, label = batch['input_ids'], batch['attention_mask'], batch['reason_input_ids'], batch['labels']
            optimizer.zero_grad()
            input_ids = prompt.to(device)
            attention_mask = prompt_attention.to(device)
            classification_labels = label.to(device)
            generation_labels = reason.to(device)
            
            loss, classification_logits, lm_logits, classification_loss, generation_loss, _ = model(input_ids, attention_mask, classification_labels, generation_labels)

            classification_logits_list.append(classification_logits.cpu())        
            classification_labels_list.append(classification_labels.cpu())
            classification_loss_list.append(classification_loss.cpu().item())
            generation_loss_list.append(generation_loss.cpu().item())


            if eval_step % 20 == 0:
                current_time = datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print("Current Time:", formatted_time)

                print(f"Step: {eval_step} / {len(validation_loader)}.")
                print(f"Loss: {loss:.3f}; classification loss: {classification_loss:.3f}; generation loss: {generation_loss:.3f}")

        classification_combined_logits = torch.cat(classification_logits_list, dim=0)
        classification_combined_labels = torch.cat(classification_labels_list, dim=0)

        classification_combined_pred = classification_combined_logits.argmax(1)

        no_injection_accuarcy, injection_accuarcy, total_accuracy = compute_accuracy(classification_combined_pred, classification_combined_labels)

        print(f"total accuracy: {total_accuracy}")
        print(f"no_injection accuarcy: {no_injection_accuarcy}")
        print(f"injection accuarcy: {injection_accuarcy}")
        print(f"classification mean loss: {np.mean(classification_loss_list)}")
        print(f"generation mean loss: {np.mean(generation_loss_list)}")
        pint_evaluate(model, tokenizer)
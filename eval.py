from transformers import AutoTokenizer
import torch
import json
from MultiTaskModel import DebertaForMultiTask

def acc_compute(model, dataset, target_label=0, name="chat"):
    bad_sample = []
    logits_list = []
    save_dict = []
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            class_logits = model.classify(sample)
            pred = class_logits.argmax().item()
            if pred != target_label:
                bad_sample.append(sample)
                logits_list.append(class_logits.cpu())
                save_dict.append({"prompt": sample, "logits": class_logits.cpu().squeeze().tolist()})

            del class_logits
            torch.cuda.empty_cache()
        
    acc = 1 - len(save_dict)/len(dataset)
    print(f"{name} set accuracy: {acc}")
    return acc

def pint_evaluate(model, tokenizer):
    with open("datasets/pint_json.json", "r") as f:
        valid_dataset = json.load(f)

    benign_set, injection_set = [], []
    chat_set, documents_set, hard_negatives_set, public_prompt_injection_set, internal_prompt_injection_set, jailbreak_set = [], [], [], [], [], []

    for sample in valid_dataset:
        if sample["label"] == False:
            if sample["category"] == "chat":
                chat_set.append(sample["text"])

            elif sample["category"] == "documents":
                documents_set.append(sample["text"])

            elif sample["category"] == "hard_negatives":
                hard_negatives_set.append(sample["text"])
            
            else:
                ValueError("Wrong Key")

            benign_set.append(sample["text"])

        elif sample["label"] == True:
            if sample["category"] == "public_prompt_injection":
                public_prompt_injection_set.append(sample["text"])

            elif sample["category"] == "internal_prompt_injection":
                internal_prompt_injection_set.append(sample["text"])

            elif sample["category"] == "jailbreak":
                jailbreak_set.append(sample["text"])

            else:
                ValueError("Wrong Key")

            injection_set.append(sample["text"])

    chat_acc = acc_compute(model, chat_set, target_label=0, name="chat")
    documents_acc = acc_compute(model, documents_set, target_label=0, name="documents")
    hard_negatives_acc = acc_compute(model, hard_negatives_set, target_label=0, name="hard_negatives")
    public_prompt_injection_acc = acc_compute(model, public_prompt_injection_set, target_label=1, name="public_prompt_injection")
    internal_prompt_injection_acc = acc_compute(model, internal_prompt_injection_set, target_label=1, name="internal_prompt_injection")
    jailbreak_acc = acc_compute(model, jailbreak_set, target_label=1, name="jailbreak")

    overall_acc = (chat_acc + documents_acc + hard_negatives_acc + public_prompt_injection_acc + internal_prompt_injection_acc + jailbreak_acc) / 6
    benign_acc = (chat_acc + documents_acc + hard_negatives_acc) / 3
    injection_acc = (public_prompt_injection_acc + internal_prompt_injection_acc + jailbreak_acc) / 3
    print(f"benign accuracy: {benign_acc}")
    print(f"injection accuracy: {injection_acc}")
    print(f"overall accuracy: {overall_acc}")
    return overall_acc


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DebertaForMultiTask('microsoft/deberta-v3-base', num_labels=2, device=device)

    resume = 'model_path'
    model.load_state_dict(resume)

    model.to(device)

    pint_evaluate(model, tokenizer)

    print("done")
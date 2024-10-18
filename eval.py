from transformers import AutoTokenizer
import torch
import json
from MultiTaskModel import DebertaForMultiTask

def acc_compute(model, target_set, target_class="benign", name="chat"):
    bad_sample = []
    logits_list = []
    save_dict = []

    if target_class == "benign":
        target_label = 0
    else:
        target_label = 1
    
    with torch.no_grad():
        for idx, sample in enumerate(target_set):
            class_logits = model.classify(sample)
            pred = class_logits.argmax().item()
            if pred != target_label:
                bad_sample.append(sample)
                logits_list.append(class_logits.cpu())
                save_dict.append({"prompt": sample, "logits": class_logits.cpu().squeeze().tolist()})
                # print(f"Prompt: {sample}\nLogits: {class_logits.cpu()}")

            del class_logits
            torch.cuda.empty_cache()
        
    acc = 1 - len(save_dict)/len(target_set)
    print(f"{name} set accuracy: {acc}")
    return acc


def pint_evaluate(model, data_path="datasets/pint_json.json"):
    with open(data_path, "r") as f:
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

    chat_acc = acc_compute(model, chat_set, target_class="benign", name="chat")
    documents_acc = acc_compute(model, documents_set, target_class="benign", name="documents")
    hard_negatives_acc = acc_compute(model, hard_negatives_set, target_class="benign", name="hard_negatives")
    public_prompt_injection_acc = acc_compute(model, public_prompt_injection_set, target_class="injection", name="public_prompt_injection")
    internal_prompt_injection_acc = acc_compute(model, internal_prompt_injection_set, target_class="injection", name="internal_prompt_injection")
    jailbreak_acc = acc_compute(model, jailbreak_set, target_class="injection", name="jailbreak")

    overall_acc = (chat_acc + documents_acc + hard_negatives_acc + public_prompt_injection_acc + internal_prompt_injection_acc + jailbreak_acc) / 6
    benign_acc = (chat_acc + documents_acc + hard_negatives_acc) / 3
    injection_acc = (public_prompt_injection_acc + internal_prompt_injection_acc + jailbreak_acc) / 3
    print(f"benign accuracy: {benign_acc}")
    print(f"injection accuracy: {injection_acc}")
    print(f"overall accuracy: {overall_acc}")
    return overall_acc, benign_acc, injection_acc

def wildguard_eval(model, data_path="datasets/wildguard"):
    benign_set = []
    with open(data_path, "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset:
        benign_set.append(valid_dataset[key]["prompt"])

    wildguard_acc = acc_compute(model, benign_set, target_class="benign", name="wildguard")
    return wildguard_acc

def BIPIA_eval(model, text_data_path="datasets/BIPIA_text", code_data_path="datasets/BIPIA_code"):
    injection_set = []
    with open(text_data_path, "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
        for context in valid_dataset[key]:
            injection_set.append(context)

    BIPIA_text_acc = acc_compute(model, injection_set, target_class="injection", name="BIPIA_text")

    injection_set = []
    with open(code_data_path, "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
        for context in valid_dataset[key]:
            injection_set.append(context)

    BIPIA_code_acc = acc_compute(model, injection_set, target_class="injection", name="BIPIA_code")

    BIPIA_overall_acc = (BIPIA_text_acc + BIPIA_code_acc) / 2
    print(f"BIPIA overall accuracy: {BIPIA_overall_acc}")

    return BIPIA_overall_acc, BIPIA_text_acc, BIPIA_code_acc

def NotInject_eval(model, one_data_path="datasets/NotInject_one", two_data_path="datasets/NotInject_two", three_data_path="datasets/NotInject_three"):
    benign_set = []
    with open(one_data_path, "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
            benign_set.append(valid_dataset[key]["instruction"])

    one_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_one")

    benign_set = []
    with open(two_data_path, "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
            benign_set.append(valid_dataset[key]["instruction"])

    two_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_two")

    benign_set = []
    with open(three_data_path, "r") as f:
        valid_dataset = json.load(f)

    for key in valid_dataset.keys():
            benign_set.append(valid_dataset[key]["instruction"])

    three_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_three")

    overall_acc = (one_acc + two_acc + three_acc) / 3
    print(f"NotInject overall accuracy: {overall_acc}")

    return overall_acc, one_acc, two_acc, three_acc

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DebertaForMultiTask('microsoft/deberta-v3-base', num_labels=2, device=device)

    resume = 'model_path'
    model.load_state_dict(resume)

    model.to(device)

    pint_acc, pint_benign_acc, pint_injection_acc = pint_evaluate(model)
    wild_acc = wildguard_eval(model)
    BIPIA_acc, BIPIA_text_acc, BIPIA_code_acc = BIPIA_eval(model)
    Notinject_acc, Notinject_one_acc, Notinject_two_acc, Notinject_three_acc = NotInject_eval(model)

    benign_acc = (pint_benign_acc + wild_acc) / 2
    injection_acc = (pint_injection_acc + BIPIA_acc) / 2
    overall_acc = (Notinject_acc + benign_acc + injection_acc) / 3

    print(f"================================ The Results ================================")
    print(f"Over-defense ACC: {Notinject_acc}")
    print(f"Benign ACC: {benign_acc}")
    print(f"Injection ACC: {injection_acc}")
    print(f"Overall ACC: {overall_acc}")

    print("done")
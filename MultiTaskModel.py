from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
import torch

class DebertaForMultiTask(nn.Module):
    def __init__(self, model_name, num_labels, device):
        super(DebertaForMultiTask, self).__init__()
        self.device = device
        self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config).to(device)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels).to(device)
        self.lm_head = nn.Linear(self.deberta.config.hidden_size, self.deberta.config.vocab_size).to(device)
        self.loss_fct = nn.CrossEntropyLoss().to(device)
        self.single_loss_fct = nn.CrossEntropyLoss(reduction='none').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    def forward(self, input_ids, attention_mask, classification_labels=None, generation_labels=None, ret_single_loss=False):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # outputs.retain_grad()
        attentions = outputs['attentions']
        grad_hidden_states = outputs['hidden_states'][0]
        last_hidden_state = outputs['last_hidden_state']
        sequence_output = last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        classification_logits = self.classifier(pooled_output)
        
        lm_logits = self.lm_head(sequence_output)
        
        loss = 0
        single_loss = 0
        classification_loss = 0
        generation_loss = torch.tensor(0).to(self.device)
        if classification_labels is not None:
            classification_loss = self.loss_fct(classification_logits.view(-1, self.classifier.out_features), classification_labels.view(-1))
            loss += classification_loss

            if ret_single_loss:
                single_loss = self.single_loss_fct(classification_logits.view(-1, self.classifier.out_features), classification_labels.view(-1))

        generation_labels = None
        if generation_labels is not None:
            generation_loss = 0.0 * self.loss_fct(lm_logits.view(-1, self.lm_head.out_features), generation_labels.view(-1))
            loss += generation_loss

        if ret_single_loss:
            return loss, classification_logits, lm_logits, classification_loss, generation_loss, single_loss            
        return loss, classification_logits, lm_logits, classification_loss, generation_loss, grad_hidden_states


    def classify(self, input_text):
        tokenzied_text = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        input_ids = tokenzied_text['input_ids'].to(self.device)
        attention_mask = tokenzied_text['attention_mask'].to(self.device)
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        sequence_output = outputs
        pooled_output = outputs[:, 0, :]

        classification_logits = self.classifier(pooled_output)
        return classification_logits

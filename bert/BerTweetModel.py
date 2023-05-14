from transformers import get_linear_schedule_with_warmup,AdamW, AutoModelForSequenceClassification
import torch
import time
from populism_detector.bert import BertUtils
import numpy as np
class BerTweetModel:

    def __init__(self, model_class="vinai/bertweet-base", num_classes=2, model_to_load=None, total_steps=-1):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_class,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        if model_to_load is not None:
            try:
                self.model.roberta.load_state_dict(torch.load(model_to_load))
                print("LOADED MODEL")
            except:
                pass

    def train(self, train_dataloader, epochs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        total_t0 = time.time()

        for epoch_i in range(0, epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = BertUtils.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                self.model.zero_grad()
                outputs = self.model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = BertUtils.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(BertUtils.format_time(time.time() - total_t0)))

    def predict(self, test_dataloader):
        self.model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        preds = []

        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            preds.append(outputs.logits)

        preds = torch.cat(preds, dim=0)
        probabilities = [el[1] for el in preds.sigmoid().cpu().numpy()]
        print(probabilities)
        labels = np.argmax(preds.cpu().numpy(), axis=1)
        return labels, probabilities
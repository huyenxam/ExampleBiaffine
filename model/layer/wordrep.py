from model.layer import CharCNN
import torch
from torch import nn
from transformers import AutoModel
# from transformers import RobertaModel


class WordRep(nn.Module):
    def __init__(self, args):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.use_char = args.use_char                       # Có sử dụng model CharCNN không
        self.xlm_roberta = AutoModel.from_pretrained(args.model_name_or_path, cache_dir="./cache/", force_download=True)
        self.num_layer_bert = args.num_layer_bert           # Số lớp bert muốn lấy
        if self.use_char:
            self.char_feature = CharCNN(hidden_dim=args.char_hidden_dim,
                                        vocab_size=args.char_vocab_size, embedding_dim=args.char_embedding_dim)

    def forward(self, input_ids, attention_mask, first_subword,  char_ids):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        bert_features = outputs[0]
        # bert_features = [bs, max_sep, 768]
        bert_features = torch.cat([torch.index_select(bert_features[i], 0, first_subword[i]).unsqueeze(0) for i in range(bert_features.size(0))], dim=0)
        # bert_features = [bs, max_sep, 768]
        if self.use_char:
            char_features = self.char_feature(char_ids)
            bert_features = torch.cat((bert_features,  char_features), dim=-1)
            # bert_features = [bs, max_sep, 768 + char_hidden_dim*2]
            return bert_features
        else:
            # bert_features = [bs, max_sep, 768]
            return bert_features


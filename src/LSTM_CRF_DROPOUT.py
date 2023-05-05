import mindspore.nn as nn
from mindspore.ops import operations as P

class LSTMCRF_dropout(nn.Cell):
    def __init__(self, config):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.dropout1 = nn.Dropout(keep_prob=config.dropout_rate)
        self.layer_norm1 = nn.LayerNorm((config.embedding_size,))
        
        self.bilstm = nn.LSTM(config.embedding_size, config.hidden_size, bidirectional=True)
        self.dropout2 = nn.Dropout(keep_prob=config.dropout_rate)
        
        self.linear = nn.Dense(config.hidden_size * 2, config.num_tags)
        self.layer_norm2 = nn.LayerNorm((config.num_tags,))
        self.crf = CRF(config.num_tags)

    def construct(self, input_ids, input_mask, labels=None):
        embedding = self.embedding(input_ids)
        embedding = self.dropout1(embedding)
        embedding = self.layer_norm1(embedding)
        
        lstm_out, _ = self.bilstm(embedding)
        lstm_out = self.dropout2(lstm_out)
        
        logits = self.linear(lstm_out)
        logits = self.layer_norm2(logits)
        
        if labels is not None:
            log_likelihood = self.crf(logits, labels, input_mask)
            return -log_likelihood
        else:
            best_paths = self.crf.decode(logits, input_mask)
            return best_paths

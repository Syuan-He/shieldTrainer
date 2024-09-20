from torch import nn, Tensor
from transformers import BertModel, BatchEncoding

from utils.DevConf import DevConf

class BERTMultiLabelClassifier(nn.Module):
    def __init__(self,
            nClass: int,
            devConf: DevConf = DevConf(),
        ):
        super(BERTMultiLabelClassifier, self).__init__()

        self.Bert = BertModel.from_pretrained('google-bert/bert-base-multilingual-cased', cache_dir="./cache/model").to(devConf.device).to(devConf.dtype)
        self.outProj = nn.Linear(768, nClass*2, device=devConf.device, dtype=devConf.dtype)
        self.activate = nn.Sigmoid().to(devConf.device).to(devConf.dtype)
        self.nClass = nClass
    
    def forward(self,
            input_ids: Tensor,
            attention_mask: Tensor,
            returnAttnWeight: bool=False
        )->tuple[Tensor, Tensor] | Tensor:

        output = self._getBertOutput(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = self.outProj(output)
        output = output.view(-1, self.nClass, 2)
        output = nn.functional.softmax(output, dim=2)

        return output
    
    def _getBertOutput(self, input_ids: Tensor, attention_mask: Tensor)->BatchEncoding:
        return self.Bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states= self.decoder.IsNeedHiddenState)
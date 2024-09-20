from torch import nn, Tensor
from transformers import DistilBertModel, BatchEncoding

from utils.DevConf import DevConf
from utils.AttnBlocksConf import AttnBlocksConf
from utils.const import BlockType
from model.BertDecoder.SentiClassifier import SentiClassifier
from model.BertDecoder.SentiClassifierWithEmbedding import SentiClassifierWithEmbedding

class CombinationModel(nn.Module):
    def __init__(self,
            nClass: int,
            attnBlocksConf: AttnBlocksConf,
            blockType: BlockType = BlockType.CROSS,
            devConf: DevConf = DevConf(),
            new_arch: bool = False
        ):
        super(CombinationModel, self).__init__()

        self.distilBert = DistilBertModel.from_pretrained('distilbert/distilbert-base-multilingual-cased', cache_dir="./cache/model").to(devConf.device).to(devConf.dtype)
        if new_arch:
            self.decoder = SentiClassifierWithEmbedding(6, nClass, attnBlocksConf, blockType, devConf)
        else:
            self.decoder = SentiClassifier(6, attnBlocksConf, blockType, devConf)
        self.outProj = nn.Linear(768, nClass*2, device=devConf.device, dtype=devConf.dtype)
        self.activate = nn.Sigmoid().to(devConf.device).to(devConf.dtype)
        self.nClass = nClass
    
    def forward(self,
            input_ids: Tensor,
            attention_mask: Tensor,
            returnAttnWeight: bool=False
        )->tuple[Tensor, Tensor] | Tensor:

        output = self._getBertOutput(input_ids=input_ids, attention_mask=attention_mask)
        output, attnWeig = self.decoder(output, returnAttnWeight=True)
        output = self.outProj(output)
        output = output.view(-1, self.nClass, 2)
        output = nn.functional.softmax(output, dim=2)
        
        if returnAttnWeight:
            return output, attnWeig
        return output
    
    def _getBertOutput(self, input_ids: Tensor, attention_mask: Tensor)->BatchEncoding:
        return self.distilBert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states= self.decoder.IsNeedHiddenState)

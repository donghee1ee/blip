from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    AutoModelForCausalLM, 
    BertModel
)
import torch.nn as nn

class FreezeInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False
    

    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        self._keys_to_ignore_on_save = set(ignore_keys)

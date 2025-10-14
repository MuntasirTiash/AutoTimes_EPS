import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)

class Model(nn.Module):
    """A model for preprocessing text data using LLaMA.

    Args:
        configs (object): An object containing the configuration parameters.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        print(self.device)
        
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(configs.llm_ckp_dir)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.vocab_size = self.llama_tokenizer.vocab_size
        self.hidden_dim_of_llama = 4096
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        """Tokenizes the input text.

        Args:
            x (str): The input text.

        Returns:
            torch.Tensor: The tokenized text as a tensor of embeddings.
        """
        output = self.llama_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        result = self.llama.get_input_embeddings()(output)
        return result

    def forecast(self, x_mark_enc):
        """Generates embeddings for the input text.

        Args:
            x_mark_enc (list[str]): A list of input strings.

        Returns:
            torch.Tensor: The embeddings for the input strings.
        """
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        text_outputs = self.llama.model(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs
    
    def forward(self, x_mark_enc):
        """Forward pass of the model.

        Args:
            x_mark_enc (list[str]): A list of input strings.

        Returns:
            torch.Tensor: The embeddings for the input strings.
        """
        return self.forecast(x_mark_enc)
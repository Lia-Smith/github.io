import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    A Transformer model for image classification tasks. Takes in a latent vectors and compute probabilities for each class.
    Args:
        input_dim (int): The dimensionality of the input features.
        num_patches (int): The number of patches (sequence length) in the input.
        output_dim (int): The dimensionality of the output (number of classes for classification).
        num_heads (int, optional): The number of attention heads in the multi-head attention mechanism.
        hidden_dim (int, optional): The dimensionality of the hidden representation.
        dropout (float, optional): Dropout rate applied in the encoder layers.
        num_layers (int, optional): The number of encoder layers in the Transformer. 
    """
    def __init__(self, input_dim: int, num_patches: int,  output_dim: int, num_heads: int = 2,
                hidden_dim: int = 64, dropout: float = 0, num_layers: int = 6):

        super(Transformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Linear Mapper - Maps input to hidden dimension
        self.linear_mapper = nn.Linear(input_dim, hidden_dim)
        
        # Class Token - Learnable parameter for classification (prepended to input sequence)
        self.class_token = nn.Parameter(torch.rand(1, hidden_dim))
        
        # Positional Encoding - Learned positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        
        # Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        Args:
            x (torch.Tensor): Input tensor of 
        Returns:
            y_hat (torch.Tensor): Output tensor
        """
        # Creates a fake sequence so that the input can be passed to the transformer
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Maps the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(x) # (batch_size, num_patches, hidden_dim)
        
        batch_size = x.shape[0]
        
        # Add class token to the input sequence
        class_tokens = self.class_token.expand(batch_size, -1).unsqueeze(1) # (batch_size, 1, hidden_dim)
        tokens = torch.cat((class_tokens, tokens), dim=1) # (batch_size, num_patches + 1, hidden_dim)

        # Add positional encoding
        output = tokens + self.pos_embedding
        
        # Transformer encoding
        for layer in self.encoder_layers:
            output = layer(output)
        
        output = output[:, 0]  # Get the class token output
        return self.classifier(output)

    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        """
        Initializes the Encoder module.
        Args:
            hidden_dim (int): The dimensionality of the hidden layers.
            num_heads (int): The number of attention heads in the multi-head attention mechanism.
            dropout (float): The dropout rate to be applied in the attention and feed-forward layers.
        """
        super().__init__()
        
        # Multihead Attention 
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim).
        Returns:
            torch.Tensor: Output tensor of the same shape as the input (batch_size, sequence_length, hidden_dim).
        """
        
        # Transpose for MultiheadAttention: (B, S, E) -> (S, B, E)
        x_t = x.transpose(0, 1)
        
        attn_output, _ = self.attention(x_t , x_t , x_t ) # MultiHead Attention
        x = self.norm1(x + attn_output.transpose(0, 1)) # Add & Norm w/ (B, S, E)
        
        ff_output = self.feed_forward(x) # Feed Forward Network
        x = self.norm2(x + ff_output) # Add & Norm
        
        return x
  
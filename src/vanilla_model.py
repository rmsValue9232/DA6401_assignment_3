import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.functional import accuracy, f1_score

class Encoder(nn.Module):
    """Encoder for the Seq2Seq model."""
    def __init__(
            self,
            input_vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            unit: str = 'gru'
        ):
        super(Encoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.unit = unit

        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)

        if unit == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        elif unit == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        elif unit == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, src: torch.Tensor, src_len: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(src)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.unit == 'lstm':
            output, (hidden, cell) = self.rnn(packed)
            return hidden, cell
        else:
            output, hidden = self.rnn(packed)
            return hidden

class Decoder(nn.Module):
    """Decoder for the Seq2Seq model."""
    def __init__(
            self,
            target_vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            unit: str = 'gru'
        ):
        super(Decoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.unit = unit

        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)

        if unit == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        elif unit == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        elif unit == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, tgt_input: torch.Tensor, hidden) -> torch.Tensor:
        embedded = self.embedding(tgt_input)
        if self.unit == 'lstm':
            h, c = hidden              # unpack
            output, (h, c) = self.rnn(embedded, (h, c))
            hidden = (h, c)            # repack for next step
        else:
            output, hidden = self.rnn(embedded, hidden)
        
        prediction = self.fc_out(output)
        return prediction

class VanillaSeq2Seq(LightningModule):
    def __init__(
            self,
            input_vocab_size: int,
            target_vocab_size: int,
            embedding_dim: int = 256,
            hidden_dim: int = 512,
            encoder_layers: int = 1,
            decoder_layers: int = 1,
            encoder_dropout: float = 0.0,
            decoder_dropout: float = 0.0,
            encoding_unit: str = 'gru',
            decoding_unit: str = 'gru',
            beam_width: int = 3,
            max_len: int = 50,
            lr: float = 1e-3,
            optimizer: str = 'adam',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            input_vocab_size,
            embedding_dim,
            hidden_dim,
            encoder_layers,
            encoder_dropout,
            encoding_unit
        )

        self.decoder = Decoder(
            target_vocab_size,
            embedding_dim,
            hidden_dim,
            decoder_layers,
            decoder_dropout,
            decoding_unit
        )
    
    def forward(self, src: torch.Tensor, src_len: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            src: Source sequences (batch_size, src_len)
            src_len: Lengths of source sequences (batch_size)
            tgt_input: Target input sequences (batch_size, tgt_len)
        Returns:
            Output predictions (batch_size, tgt_len, target_vocab_size)
        """
        hidden = self.encoder(src, src_len)
        logits = self.decoder(tgt_input, hidden)
        return logits
    
    def configure_optimizers(self):
        """Configure the optimizer."""
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)    
        elif self.hparams.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
        return optimizer
    
    def _compute_loss_and_metrics(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the loss for a batch.
        Args:
            batch: Batch of data
        Returns:
            Loss value
        """
        src_input = batch['src_input']
        src_len = batch['src_len']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']

        logits = self(src_input, src_len, tgt_input)
        
        # Reshape logits and target for loss computation
        logits = logits.view(-1, self.hparams.target_vocab_size)
        tgt_output = tgt_output.reshape(-1)

        loss = F.cross_entropy(logits, tgt_output, ignore_index=0) # Ignore padding index

        preds_flat = torch.argmax(logits, dim=-1)
        mask = (tgt_output != 0)

        acc = accuracy(preds_flat[mask], tgt_output[mask], num_classes=self.hparams.target_vocab_size, task = 'multiclass')
        f1 = f1_score(preds_flat[mask], tgt_output[mask], num_classes=self.hparams.target_vocab_size, task = 'multiclass', average='macro')
        
        return loss, acc, f1
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.
        Args:
            batch: Batch of data
            batch_idx: Batch index
        Returns:
            Loss value
        """
        batch_size = batch['src_input'].size(0)
        loss, acc, f1 = self._compute_loss_and_metrics(batch)

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for the model.
        Args:
            batch: Batch of data
            batch_idx: Batch index
        Returns:
            Loss value
        """
        batch_size = batch['src_input'].size(0)
        loss, acc, f1 = self._compute_loss_and_metrics(batch)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        return loss
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step for the model.
        Args:
            batch: Batch of data
            batch_idx: Batch index
        Returns:
            Loss value
        """
        batch_size = batch['src_input'].size(0)
        loss, acc, f1 = self._compute_loss_and_metrics(batch)

        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("test_f1", f1, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        return loss
    
    def beam_search(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
            sos_idx: int,
            eos_idx: int,
            max_len: int,
            beam_width: int = 3,
    ) -> tuple[list[int], float]:
        """
        Perform beam search decoding.
        Args:
            src: Source sequence (1, src_len)
            src_len: Lengths of source sequence (1,)
            sos_idx: Start of sequence index
            eos_idx: End of sequence index
            max_len: Maximum length of output sequences
            beam_width: Beam width for beam search
        Returns:
            Best sequence and its score
        """
        # Encode source to get initial hidden state
        hidden = self.encoder(src, src_len)

        # Initialize beam search with SOS token
        sequences = [([sos_idx], 0.0, hidden)]
        completed = []

        # Iteratively expand
        for _ in range(max_len):
            all_candidates = []
            for seq, score, h in sequences:
                # If EOS was generated, add to completed
                if seq[-1] == eos_idx:
                    completed.append((seq, score, h))
                    continue

                # Prepare last token as input
                last_token = torch.tensor([[seq[-1]]], device=src.device)
                embedded = self.decoder.embedding(last_token)

                if self.decoder.unit == 'rnn':
                    output, h = self.decoder.rnn(embedded, h)
                elif self.decoder.unit == 'gru':
                    output, h = self.decoder.rnn(embedded, h)
                elif self.decoder.unit == 'lstm':
                    output, (h, cell) = self.decoder.rnn(embedded, h)
                
                logits = self.decoder.fc_out(output.squeeze(1))
                log_probs = F.log_softmax(logits, dim=-1)

                # Get top k candidates
                top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width)
                for k in range(beam_width):
                    next_id = top_k_indices[0, k].item()
                    new_score = score + top_k_log_probs[0, k].item()
                    all_candidates.append((seq + [next_id], new_score, h))

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        completed.extend(sequences)
        best_seq, best_score, h = max(completed, key=lambda x: x[1])
        return best_seq, best_score
    
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> list[int]:
        """
        Prediction step for the model.
        Args:
            batch: Batch of data
            batch_idx: Batch index
        Returns:
            Predicted sequence
        """
        src_input = batch['src_input']
        src_len = batch['src_len']
        preds = []
        
        for i in range(src_input.size(0)):
            best_seq, best_score = self.beam_search(
                src_input[i:i+1],
                src_len[i:i+1],
                sos_idx=1,
                eos_idx=2,
                max_len=self.hparams.max_len,
                beam_width=self.hparams.beam_width
            )
            preds.append(best_seq)
        
        return preds
        


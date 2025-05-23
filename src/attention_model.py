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
            outputs, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return outputs, (hidden, cell)
        else:
            output, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return outputs, hidden

class DotAttention(nn.Module):
    """Dot-product attention mechanism."""
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs, mask = None):
        """
        decoder_hidden: (1, batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, hidden_dim)
        """
        # Reshape decoder hidden state for bmm
        dec = decoder_hidden.permute(1, 2, 0)

        # Compute attention scores
        scores = torch.bmm(encoder_outputs, dec).squeeze(2)  # (batch_size, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights # (batch_size, hidden_dim), (batch_size, seq_len)

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
        
        self.attention = DotAttention()

        self.fc_out = nn.Linear(hidden_dim * 2, target_vocab_size)
    
    def forward(self, tgt: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask = None):
        embedded = self.embedding(tgt)

        if self.unit == 'lstm':
            h, c = hidden              # unpack
            output, (h, c) = self.rnn(embedded, (h, c))
            hidden = (h, c)            # repack for next step
        else:
            output, hidden = self.rnn(embedded, hidden)
        

        logits = []
        attentions = []

        for t in range(output.size(1)):
            # Get the current time step's output
            output_t = output[:, t, :].unsqueeze(0) # (1, batch_size, hidden_dim)
            # Attend
            context, attn_weights = self.attention(output_t, encoder_outputs, mask)
            # Concat
            combined = torch.cat([output_t.squeeze(0), context], dim=1) # (batch_size, hidden_dim * 2)
            # Project
            logit_t = self.fc_out(combined) # (batch_size, target_vocab_size)
            logits.append(logit_t.unsqueeze(1))
            attentions.append(attn_weights.unsqueeze(1))
        
        logits = torch.cat(logits, dim=1)
        attentions = torch.cat(attentions, dim=1)
        
        return logits, hidden, attentions

class AttentionSeq2Seq(LightningModule):
    """Seq2Seq model with attention mechanism."""
    def __init__(
            self,
            input_vocab_size: int,
            target_vocab_size: int,
            embedding_dim: int = 256,
            hidden_dim: int = 512,
            encoder_layers: int = 2,
            decoder_layers: int = 2,
            encoder_dropout: float = 0.0,
            decoder_dropout: float = 0.0,
            encoding_unit: str = 'gru',
            decoding_unit: str = 'gru',
            max_len: int = 50,
            beam_width: int = 3,
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
    
    def forward(self, src: torch.Tensor, src_len: torch.Tensor, tgt: torch.Tensor):
        encoder_outputs, encoder_hidden = self.encoder(src, src_len)
        
        # Mask for padding
        mask = (src != 0).to(src.device)

        logits, hidden, attention_weights = self.decoder(tgt, encoder_hidden, encoder_outputs, mask)
        
        return logits, hidden, attention_weights
    
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

        logits, hidden, attention_weights = self(src_input, src_len, tgt_input)
        
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

        encoder_outputs, encoder_hidden = self.encoder(src, src_len)
        enc_seq_len = encoder_outputs.size(1)
        
        # Mask for padding
        mask = (
            torch.arange(enc_seq_len, device=src.device)
            .unsqueeze(0)
            < src_len.unsqueeze(1)
        )

        beams = [([sos_idx], 0.0, encoder_hidden, [])]
        completed = []

        for _ in range(max_len):
            all_candidates = []
            for seq, score, h, attn_list in beams:
                if seq[-1] == eos_idx:
                    completed.append((seq, score, h, attn_list))
                    continue

                last_token = torch.tensor([[seq[-1]]], device=src.device)
                embedded = self.decoder.embedding(last_token)  # (1, 1, embedding_dim)
                
                if self.decoder.unit == 'rnn':
                    output, h_new = self.decoder.rnn(embedded, h)
                elif self.decoder.unit == 'gru':
                    output, h_new = self.decoder.rnn(embedded, h)
                elif self.decoder.unit == 'lstm':
                    output, (h_new, cell) = self.decoder.rnn(embedded, h)
                
                context, attn_weights = self.decoder.attention(output.permute(1, 0, 2), encoder_outputs, mask)

                dec_t = output.squeeze(1)  # (1, hidden_dim)
                combined = torch.cat([dec_t, context], dim=1)
                logits = self.decoder.fc_out(combined)
                log_probs = F.log_softmax(logits, dim=1)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                for k in range(beam_width):
                    candidate = (
                        seq + [topk_indices[0, k].item()],
                        score + topk_log_probs[0, k].item(),
                        h_new,
                        attn_list + [attn_weights.squeeze(0).cpu()]
                    )
                    all_candidates.append(candidate)
            
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        completed.extend(beams)
        best_seq, best_score, h, best_attn_list = max(completed, key=lambda x: x[1])

        best_attns = torch.stack(best_attn_list, dim=0)  # (seq_len, src_len)
        return best_seq, best_score, best_attns
    
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
        attns = []
        for i in range(src_input.size(0)):
            best_seq, _, best_attns = self.beam_search(
                src_input[i:i+1],
                src_len[i:i+1],
                sos_idx=1,
                eos_idx=2,
                max_len=self.hparams.max_len,
                beam_width=self.hparams.beam_width
            )
            preds.append(best_seq)
            attns.append(best_attns)
        
        return preds, attns
    

    

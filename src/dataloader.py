from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import csv
import torch

class CharVocab:
    """A simple character vocabulary class."""
    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'

    def __init__(self):
        self.idx2char: list[str] = [self.PAD, self.SOS, self.EOS]
        self.char2idx: dict[str, int] = {c: i for i, c in enumerate(self.idx2char)}
    
    def build(self, sequences: list[str]):
        """Build vocabulary from a list of strings, adding each new unique character."""
        for seq in sequences:
            for char in seq:
                if char not in self.char2idx:
                    self.char2idx[char] = len(self.idx2char)
                    self.idx2char.append(char)
    
    def encode(self, sequence: str) -> list[int]:
        """Encode a string to a list of indices, including SOS and EOS"""

        return [self.char2idx[self.SOS]] + [self.char2idx[char] for char in sequence] + [self.char2idx[self.EOS]]
    
    def decode(self, indices: list[int]) -> str:
        """Decode a list of indices (ignoring PAD, SOS, and EOS) back to a string."""
        chars = []
        for idx in indices:
            ch = self.idx2char[idx]
            if ch in (self.PAD, self.SOS, self.EOS):
                continue
            chars.append(ch)
        
        return ''.join(chars)
    
    @property
    def size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.idx2char)

class DakshinaDataset(Dataset):
    """A dataset class for the Dakshina dataset."""
    def __init__(
            self,
            data_file: str,
            input_vocab: CharVocab,
            target_vocab: CharVocab
        ):
        self.pairs: list[tuple[str, str]] = []

        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter = '\t')
            for row in reader:
                src, tgt, _ = row
                self.pairs.append((src, tgt))
        
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample from the dataset."""
        src, tgt = self.pairs[idx]
        src_encoded = self.input_vocab.encode(src)
        tgt_encoded = self.target_vocab.encode(tgt)
        
        return {
            'src': torch.tensor(src_encoded, dtype=torch.long),
            'tgt': torch.tensor(tgt_encoded, dtype=torch.long)
        }

class DakshinaDataModule(LightningDataModule):
    """A LightningDataModule for the Dakshina dataset."""
    def __init__(
            self,
            train_file: str,
            val_file: str,
            test_file: str,
            batch_size: int = 32,
            num_workers: int = 2,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize vocabularies, built in setup step
        self.input_vocab = CharVocab()
        self.target_vocab = CharVocab()
    
    def setup(self, stage: str = None):
        """Setup the datasets and vocabularies."""
        inputs, outputs = [], []

        with open(self.train_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter = '\t')
            for row in reader:
                src, tgt, _ = row
                inputs.append(src)
                outputs.append(tgt)
        
        # Build vocabularies from training data
        self.input_vocab.build(inputs)
        self.target_vocab.build(outputs)

        # Create datasets
        self.train_dataset = DakshinaDataset(self.train_file, self.input_vocab, self.target_vocab)
        self.val_dataset = DakshinaDataset(self.val_file, self.input_vocab, self.target_vocab)
        self.test_dataset = DakshinaDataset(self.test_file, self.input_vocab, self.target_vocab)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Pads source and target sequences to the maximum length in the batch.
        Returns dict with src_input, src_len, tgt_input, tgt_len, tgt_output.
        """

        src_seqs = [item['src'] for item in batch]
        tgt_seqs = [item['tgt'] for item in batch]

        src_lens = torch.tensor([len(seq) for seq in src_seqs], dtype=torch.long)
        tgt_lens = torch.tensor([len(seq) for seq in tgt_seqs], dtype=torch.long)

        # Pad sequences
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_seqs, batch_first=True, padding_value=self.input_vocab.char2idx[CharVocab.PAD]
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_seqs, batch_first=True, padding_value=self.target_vocab.char2idx[CharVocab.PAD]
        )

        # For teacher forcing, inputs are all tokens except last, outputs are all except first
        tgt_input = tgt_padded[:, :-1]
        tgt_output= tgt_padded[:, 1: ]

        return {
            'src_input': src_padded,
            'src_len': src_lens,
            'tgt_input': tgt_input,
            'tgt_len': tgt_lens - 1,
            'tgt_output': tgt_output
        }
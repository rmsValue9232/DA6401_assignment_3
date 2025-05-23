from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import dataloader
import vanilla_model
import attention_model

# input_vocab_size: int,
# target_vocab_size: int,
# embedding_dim: int = 256,
# hidden_dim: int = 512,
# encoder_layers: int = 2,
# decoder_layers: int = 2,
# encoder_dropout: float = 0.0,
# decoder_dropout: float = 0.0,
# encoding_unit: str = 'gru',
# decoding_unit: str = 'gru',
# max_len: int = 50,
# beam_width: int = 3,
# lr: float = 1e-3,
# optimizer: str = 'adam',

# input embedding size: 16, 32, 64, 256, ...
# number of encoder layers: 1, 2, 3
# number of decoder layers: 1, 2, 3
# hidden layer size: 16, 32, 64, 256, ...
# cell type: RNN, GRU, LSTM
# dropout: 20%, 30% (btw, where will you add dropout? you should read up a bit on this)
# beam search in decoder with different beam sizes:


params = dict()

params['embedding_dim'] = {
    'values': [32, 64, 128, 256, 512]
}
params['hidden_dim'] = {
    'values': [32, 64, 128, 256, 512]
}
params['layers_count'] = {
    'values': [1, 2, 3]
}
params['encoder_dropout'] = {
    'values': [0.0, 0.1, 0.2, 0.3]
}
params['decoder_dropout'] = {
    'values': [0.0, 0.1, 0.2, 0.3]
}
params['encoding_unit'] = {
    'values': ['rnn', 'gru', 'lstm']
}
params['decoding_unit'] = {
    'values': ['rnn', 'gru', 'lstm']
}
params['beam_width'] = {
    'values': [1, 2, 3, 4, 5]
}
params['lr'] = {
    'values': [1e-3, 1e-4, 1e-5]
}
params['optimizer'] = {
    'values': ['adam', 'sgd', 'rmsprop']
}
params['batch_size'] = {
    'values': [16, 32, 64, 128]
}

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': params
}

class KiteRunner:
    """Return a WandB train function based on the model type."""
    def __init__(
            self,
            train_file_path: str,
            val_file_path: str,
            test_file_path: str,
            model = 'vanilla' | 'attention'
        ):
        self.modeltype = model
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path
    
    def trainfunc(self, params: dict = None):
        """Train the model using current parameters."""
        run_name = self.modeltype
        if params is None:
            wandb.init()
        else:
            wandb.init(config = params)
        
        run_name += f"_ENC{wandb.config['encoding_unit']}"
        run_name += f"_DEC{wandb.config['decoding_unit']}"
        run_name += f"_EMB{wandb.config['embedding_dim']}"
        run_name += f"_HID{wandb.config['hidden_dim']}"
        run_name += f"_BEA{wandb.config['beam_width']}"

        wandb.run.name = run_name
        
        data_module = dataloader.DakshinaDataModule(
            train_file = self.train_file_path,
            val_file = self.val_file_path,
            test_file = self.test_file_path,    
            batch_size = wandb.config['batch_size'],
            num_workers = 2
        )
        data_module.setup()

        wandb_logger = WandbLogger(save_dir='wandb_logs', log_model=True)

        wandb_logger.log_hyperparams(wandb.config)

        if self.modeltype == 'vanilla':
            model = vanilla_model.VanillaSeq2Seq(
                input_vocab_size = data_module.input_vocab.size,
                target_vocab_size = data_module.target_vocab.size,
                embedding_dim = wandb.config['embedding_dim'],
                hidden_dim = wandb.config['hidden_dim'],
                encoder_layers = wandb.config['layers_count'],
                decoder_layers = wandb.config['layers_count'],
                encoder_dropout = wandb.config['encoder_dropout'],
                decoder_dropout = wandb.config['decoder_dropout'],
                encoding_unit = wandb.config['encoding_unit'],
                decoding_unit = wandb.config['decoding_unit'],
                max_len = 50,
                beam_width = wandb.config['beam_width'],
                lr = wandb.config['lr'],
                optimizer = wandb.config['optimizer']
            )
        else:
            model = attention_model.AttentionSeq2Seq(
                input_vocab_size = data_module.input_vocab.size,
                target_vocab_size = data_module.target_vocab.size,
                embedding_dim = wandb.config['embedding_dim'],
                hidden_dim = wandb.config['hidden_dim'],
                encoder_layers = wandb.config['encoder_layers'],
                decoder_layers = wandb.config['decoder_layers'],
                encoder_dropout = wandb.config['encoder_dropout'],
                decoder_dropout = wandb.config['decoder_dropout'],
                encoding_unit = wandb.config['encoding_unit'],
                decoding_unit = wandb.config['decoding_unit'],
                max_len = 50,
                beam_width = wandb.config['beam_width'],
                lr = wandb.config['lr'],
                optimizer = wandb.config['optimizer']
            )
        
        wandb_logger.watch(model, log='all', log_graph=True)

        checkpoint_callback = ModelCheckpoint(
            monitor = 'val_acc',
            mode = 'max',
            dirpath = 'checkpoints',
            filename = f"{self.modeltype}-{{epoch:02d}}-{{val_acc:.2f}}"
        )

        early_stopping_callback = EarlyStopping(
            monitor = 'val_acc',
            patience = 5,
            mode = 'max'
        )

        trainer = Trainer(
            max_epochs = 10,
            devices = 1,
            logger = wandb_logger,
            callbacks = [checkpoint_callback, early_stopping_callback],
            log_every_n_steps = 10,
        )

        trainer.fit(
            model,
            data_module,
        )

        trainer.test(
            model,
            data_module,
        )

        wandb.finish()
        
def sweeper(
        sweep_config: dict,
        trainfunc: callable,
        max_runs: int = 10,
    ):
    """Run a sweep using the given config and train function."""
    sweep_id = wandb.sweep(
        sweep = sweep_config,
        project = 'da6401_assignment_3'
    )
    wandb.agent(
        sweep_id,
        function = trainfunc,
        count = max_runs
    )
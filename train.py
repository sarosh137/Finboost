import argparse, os, torch, numpy as np
from src.utils import load_config, get_logger, ensure_dir
from src.data_loader import load_all_symbols
from src.features import add_technical_indicators, build_feature_matrix
from src.models import FinboostModel

logger = get_logger(__name__)

def prepare_train_data(cfg):
    symbols = load_all_symbols(cfg['data']['sample_data_dir'])
    # For starter: take one symbol and create a small dataset
    first_sym = list(symbols.keys())[0]
    df = symbols[first_sym]
    df = add_technical_indicators(df)
    X = build_feature_matrix(df, seq_len=cfg['data']['seq_len'])
    # Dummy targets: next return and reversal label (toy)
    y_ret = np.random.randn(len(X),1).astype('float32')
    y_rev = (np.random.rand(len(X),1) > 0.95).astype('float32')
    return X.astype('float32'), y_ret, y_rev

def train(cfg, dry_run=False):
    X, y_ret, y_rev = prepare_train_data(cfg)
    batch = cfg['training']['batch_size']
    input_dim = X.shape[-1]
    model = FinboostModel(input_dim, hidden_dim=64, levels=3)
    if dry_run:
        logger.info('Dry run: model instantiated. X shape: %s', X.shape)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    import torch.nn as nn
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    model.train()
    for epoch in range(cfg['training']['epochs']):
        # simple batch loop
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch])
            yb = torch.tensor(y_ret[i:i+batch])
            yr = torch.tensor(y_rev[i:i+batch])
            out = model(xb)
            loss = mse(out['next_candle'], yb) + 0.5*bce(out['reversal'], yr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info('Epoch %d done', epoch+1)
    logger.info('Training complete (toy). Save model not implemented in starter.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/settings.yaml')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run)

from __future__ import annotations
import argparse, json
from pathlib import Path

from psd_snn.config.specs import load_experiment_config, to_sanitized_dict
from psd_snn.models.factory import build_model


def _make_data(torch, cfg, n, seed=0):
    g=torch.Generator().manual_seed(seed)
    if cfg.model.topology.kind in {'vgg','resnet'}:
        x=torch.randn(n, 16, 1, 8, 8, generator=g)
    else:
        x=torch.randn(n, 16, cfg.model.topology.input_dim, generator=g)
    y=torch.randint(0, cfg.model.topology.output_dim, (n,), generator=g)
    return x,y


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--run_id', default='train_run')
    args=ap.parse_args(argv)

    import torch
    cfg=load_experiment_config(args.config)
    model=build_model(cfg.model).to(args.device)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn=torch.nn.CrossEntropyLoss()
    x,y=_make_data(torch, cfg, 32, seed=args.seed)
    model.train()
    for _ in range(args.epochs):
        for i in range(0, len(x), args.batch_size):
            xb=x[i:i+args.batch_size].to(args.device)
            yb=y[i:i+args.batch_size].to(args.device)
            logits=model(xb)
            loss=loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt=out/'checkpoint_epoch_1.pt'
    payload={
        'state_dict': model.state_dict(),
        'config': to_sanitized_dict(cfg),
        'checkpoint_epoch': 1,
        'metadata': {
            'run_id': args.run_id,
            'model': {'topology.kind': cfg.model.topology.kind},
            'train': {'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed, 'synthetic': bool(args.synthetic)}
        }
    }
    torch.save(payload, ckpt)
    print(json.dumps({'checkpoint': str(ckpt)}))


if __name__=='__main__':
    main()

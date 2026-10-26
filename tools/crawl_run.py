"""Fetch one W&B run with metrics, system events and launcher-independent logs."""
import argparse
import json

try:
    from .crawl_storage import new_snapshot, write_json
except ImportError:
    from crawl_storage import new_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='entity/project/run_id')
    args = parser.parse_args()
    import wandb
    run = wandb.Api(timeout=30).run(args.run)
    folder = new_snapshot('wandb', run.id, entity=run.entity, project=run.project,
                          run_id=run.id, url=run.url, run_created_at=str(run.created_at))
    write_json(folder / 'config.json', dict(run.config))
    write_json(folder / 'summary.json', dict(run.summary))
    write_json(folder / 'history.json', list(run.scan_history()))
    write_json(folder / 'events.json', run.history(stream='events', samples=10000, pandas=False))
    write_json(folder / 'artifacts.json', [
        {'name': a.name, 'type': a.type, 'size': a.size, 'metadata': a.metadata}
        for a in run.logged_artifacts()
    ])
    for f in run.files():
        if f.name in ('output.log', 'wandb-metadata.json'):
            f.download(root=str(folder), replace=False)
    print(folder)
    print('System events use the W&B history API (up to 10000 samples); W&B output.log may omit launcher errors.')


if __name__ == '__main__':
    main()

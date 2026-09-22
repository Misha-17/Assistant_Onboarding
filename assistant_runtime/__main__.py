"""Keep the core CLI; compose the optional citation service for the web UI."""
from __future__ import annotations
import sys


def main(argv=None):
    from sisu_reader import cli
    from sisu_reader.config import Config
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        values = ['ui']
    args = cli._parser().parse_args(values)
    if args.command != 'ui':
        return cli.main(values)
    try:
        config = Config.load()
        if args.workspace:
            config = config.with_overrides(workspace_dir=args.workspace.resolve())
        if args.model:
            config = config.with_overrides(model=args.model.strip())
        if args.user:
            if not args.user.strip():
                raise ValueError('--user cannot be empty')
            config = config.with_overrides(principal_id=args.user.strip())
        from .audit_app import run_web
        run_web(config, open_browser=not args.no_browser)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

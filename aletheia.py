#!/usr/bin/env python3
from pathlib import Path

import click
from click_repl import register_repl, repl
from prompt_toolkit.history import FileHistory

import aletheialib.options as options


@click.group()
@click.option('--batch/--no-batch', default=False, help="Run a subcommand in batch mode if possible.")
@click.option('--verbose/--no-verbose', default=False, help="Prints more information.")
@click.pass_context
def main(ctx, batch, verbose):
    ctx.ensure_object(dict)
    ctx.obj['batch'] = batch
    ctx.obj['verbose'] = verbose


main.add_command(options.auto.auto)
main.add_command(options.auto.dci)
main.add_command(options.brute_force.brute_force)
main.add_command(options.calibration.launch)
main.add_command(options.embsim.embsim)
main.add_command(options.feaext.feaext)
main.add_command(options.ml.ml)
main.add_command(options.structural.structural)
main.add_command(options.tools.tools)

history_path = Path('.local/share/aletheia/history.txt')


@main.command()
def clear():
    """Clears the command history and caches."""
    if history_path.exists():
        history_path.unlink()


@main.command('repl')
def main_repl():
    """Runs this programm in REPL mode."""
    history_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_kwargs = {
        'history': FileHistory(history_path),
    }
    repl(click.get_current_context(), prompt_kwargs=prompt_kwargs)


if __name__ == "__main__":
    main()

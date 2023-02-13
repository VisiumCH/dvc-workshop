"""Module for training the model."""

from pathlib import Path

import typer


def main(preprocessed_file: Path = typer.Option(...), output_file: Path = typer.Option(...)) -> None:
    """Main function, print hello message."""
    print("Reading message from preprocessing:")
    print(preprocessed_file.read_text())

    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text("Hello from training!")


if __name__ == "__main__":
    typer.run(main)

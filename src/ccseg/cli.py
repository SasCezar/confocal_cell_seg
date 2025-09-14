import logging
from pathlib import Path

from tqdm import tqdm
import typer

from ccseg.convert import lif_image_to_imagej_tiff_raw
from ccseg.export import dump
from ccseg.io.factory import ImageReaderFactory

from ccseg.segmenter import segment_3d

from readlif.reader import LifFile
from ccseg.settings import Settings
from ccseg.viz import visualize

app = typer.Typer(add_completion=False)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    )


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml")):
    cfg = Settings.from_yaml(config)
    setup_logging(cfg.runtime.log_level)
    out_dir = cfg.data.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = cfg.data.input_dir
    files = input_dir.glob("**.tif*")
    for input in tqdm(files):
        logging.info(f"Processing image {input}")
        reader = ImageReaderFactory().get_reader(input)
        channels, metadata = reader.read(input)
        if not channels:
            logging.info(f"Skippint image {input} - Not enough channels.")
            continue

        logging.info(f"Image metadata {metadata} - Shape channel 0 {channels[0].shape}")

        df, stages = segment_3d(cfg=cfg.segmentation, channels=channels, metadata=metadata)

        csv_out = out_dir / "centroids" / f"{input.stem}.csv"
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out)
        if cfg.export.enable:
            logging.info("Creating layers dumb")
            dump(stages=stages, keys=cfg.export.layers, image_name=input.stem, out_dir=out_dir)

        if cfg.visualization.enable:
            logging.info("Visualization")
            visualize(stages=stages, keys=cfg.visualization.layers)


@app.command()
def convert(config: Path = typer.Option(..., "--config", "-c", help="Path to convert config.yaml")):
    cfg = Settings.from_yaml(config)
    lif_folder = cfg.convert.lif_dir
    tiff_out = cfg.convert.tiff_out
    tiff_out.mkdir(exist_ok=True)
    liff_files = list(lif_folder.glob("**/*.lif"))
    for file in tqdm(liff_files):
        lif = LifFile(str(file))
        for i, img in enumerate(lif.get_iter_image()):
            file_out = tiff_out / f"{file.stem} - {img.name}.tiff"
            lif_image_to_imagej_tiff_raw(img=img, out_path=file_out)


if __name__ == "__main__":
    app()

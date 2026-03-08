"""
map_decoder.py
================

This script demonstrates how to decode the proprietary `Map.dat` format
used by the Vorlia client.  From analysing `main.h`, `FileHandling.cpp`
and `minimap.cpp`, we know the map file contains a 25‑byte header
followed by two complete layers of 4 096 chunk definitions and then a
512×512 chunk map.  Each chunk is a 16×16 grid of unsigned 16‑bit
values.  The low 10 bits of each value encode a tile index (0–1023),
which corresponds to an entry in the `Tiles` array loaded from
`data02.dat`.  The high bits encode flags for flipping, layering or
blocked/unblocked status.  This script demonstrates how to parse the
file and reconstruct a small region of the world.

The script relies on Pillow (PIL) to load the tile sheets and to
compose output images.  Install Pillow via:

    pip install pillow

Usage:

    python map_decoder.py

It will read `map.dat`, `tiles.png` and `objects.png` in the current
directory and produce a small preview image (`preview.png`) showing
several chunks of the decoded map.  The code can easily be extended
to render the full map, but beware that the complete world is
8192×8192 tiles, which results in a very large image (~131 k×131 k
pixels) and will consume significant memory.
"""

import struct
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "This script requires Pillow. Install it with `pip install pillow` and run again."
    )


TILE_SIZE = 16
NUM_CHUNKS = 4096
NUM_LAYERS = 2
CHUNK_DIM = 16  # tiles per chunk side
CHUNK_MAP_SIZE = 512  # number of chunks per side in the world


def load_tile_sheet(path: Path) -> List[Image.Image]:
    """
    Load a tile sheet from a PNG/BMP/VDF/DAT file and return a list of 16×16 RGBA tile images.

    The converter in the original source writes tile graphics to a
    ``gdataXX.vdf`` file with a very simple header: four bytes of
    ``height << 2``, four bytes of random data, four bytes of ``width << 2``,
    followed by ``height * width * 3`` bytes of RGB pixel data.  If
    ``path`` ends with ``.vdf`` or ``.dat`` we assume this format and
    decode it accordingly.  Otherwise the file is opened via Pillow
    directly (e.g. for ``.png`` or ``.bmp`` files).
    """
    suffix = path.suffix.lower()
    if suffix in (".vdf", ".dat"):  # decode custom binary tile sheet
        with path.open("rb") as f:
            # Read and decode header
            height4 = struct.unpack("<I", f.read(4))[0]
            height = height4 >> 2
            _random = f.read(4)  # random value, ignore
            width4 = struct.unpack("<I", f.read(4))[0]
            width = width4 >> 2
            # Read pixel data (24‑bit RGB)
            data = f.read(height * width * 3)
            if len(data) != height * width * 3:
                raise ValueError(f"Unexpected length for pixel data in {path}")
            # Create an RGB image and then convert to RGBA
            img = Image.frombytes("RGB", (width, height), data)
            img = img.convert("RGBA")
    else:
        # Assume a standard image (PNG/BMP/JPEG) that Pillow can read
        img = Image.open(path).convert("RGBA")
    # Split into 16×16 tiles
    tiles: List[Image.Image] = []
    tiles_x = img.width // TILE_SIZE
    tiles_y = img.height // TILE_SIZE
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = img.crop(
                (tx * TILE_SIZE, ty * TILE_SIZE, (tx + 1) * TILE_SIZE, (ty + 1) * TILE_SIZE)
            )
            tiles.append(tile)
    return tiles


def read_map(path: Path) -> Tuple[List[List[List[List[int]]]], List[List[int]]]:
    """Parse the Map.dat file and return (chunks, chunk_map).

    chunks is a list of NUM_LAYERS elements, each a list of 4096 chunks.
    Each chunk is a 16×16 list of 16‑bit integers.
    chunk_map is a 512×512 grid of 16‑bit integers indexing into the chunk list.
    """
    with path.open("rb") as f:
        header = f.read(25)
        # Parse two layers of chunks
        layers: List[List[List[List[int]]]] = []
        for layer_idx in range(NUM_LAYERS):
            layer_chunks: List[List[List[int]]] = []
            for chunk_id in range(NUM_CHUNKS):
                chunk: List[List[int]] = []
                for row in range(CHUNK_DIM):
                    row_vals = []
                    for col in range(CHUNK_DIM):
                        raw = f.read(2)
                        if len(raw) != 2:
                            raise ValueError(
                                f"Unexpected EOF while reading chunk {chunk_id} layer {layer_idx}"
                            )
                        val = struct.unpack("<H", raw)[0]
                        row_vals.append(val)
                    chunk.append(row_vals)
                layer_chunks.append(chunk)
            layers.append(layer_chunks)
        # Read chunk map
        chunk_map: List[List[int]] = []
        for row in range(CHUNK_MAP_SIZE):
            row_vals = []
            for col in range(CHUNK_MAP_SIZE):
                raw = f.read(2)
                if len(raw) != 2:
                    raise ValueError("Unexpected EOF while reading chunk map")
                val = struct.unpack("<H", raw)[0]
                row_vals.append(val)
            chunk_map.append(row_vals)
    return layers, chunk_map


def decode_tile_value(val: int) -> Tuple[int, dict]:
    """Split a 16‑bit tile value into (tile_id, flags).

    According to `main.h`, there are 1024 tile definitions.  We assume
    the lower 10 bits hold the tile ID (0–1023) and the upper 6 bits
    encode flags.  This function returns the tile ID and a dict of
    flags that could be used to flip the tile or handle blocked cells.
    Note: You may need to adjust the bit masks if testing shows a
    different layout.
    """
    tile_id = val & 0x03FF  # 10 bits
    flags_val = (val >> 10) & 0x003F
    flags = {
        "flip_h": bool(flags_val & 0x01),
        "flip_v": bool(flags_val & 0x02),
        "block": bool(flags_val & 0x04),
        # Additional flag bits can be interpreted here
    }
    return tile_id, flags


def render_preview(
    layers: List[List[List[List[int]]]],
    chunk_map: List[List[int]],
    ground_tiles: List[Image.Image],
    object_tiles: List[Image.Image],
    out_path: Path,
    chunks_w: int = 4,
    chunks_h: int = 4,
):
    """Render a small preview of the map and save it as a PNG.

    The preview will be chunks_w×chunks_h chunks in size.  Each chunk
    covers 16×16 tiles.  Only the bottom layer is drawn for simplicity,
    but you can extend this to draw multiple layers.
    """
    width = chunks_w * CHUNK_DIM
    height = chunks_h * CHUNK_DIM
    out = Image.new("RGBA", (width * TILE_SIZE, height * TILE_SIZE))
    for cy in range(chunks_h):
        for cx in range(chunks_w):
            chunk_id = chunk_map[cy][cx]
            chunk = layers[0][chunk_id]  # draw only ground layer
            for ty in range(CHUNK_DIM):
                for tx in range(CHUNK_DIM):
                    val = chunk[ty][tx]
                    tile_id, flags = decode_tile_value(val)
                    # Choose from ground or object tiles based on a flag bit
                    # Here we simply use ground_tiles for all values; adjust as needed
                    if tile_id < len(ground_tiles):
                        tile = ground_tiles[tile_id]
                    elif tile_id - len(ground_tiles) < len(object_tiles):
                        tile = object_tiles[tile_id - len(ground_tiles)]
                    else:
                        # Unknown tile ID, draw magenta placeholder
                        tile = Image.new(
                            "RGBA", (TILE_SIZE, TILE_SIZE), (255, 0, 255, 255)
                        )
                    # Apply flips if set
                    if flags["flip_h"]:
                        tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
                    if flags["flip_v"]:
                        tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
                    out.paste(
                        tile,
                        (
                            (cx * CHUNK_DIM + tx) * TILE_SIZE,
                            (cy * CHUNK_DIM + ty) * TILE_SIZE,
                        ),
                    )
    out.save(out_path)
    print(f"Saved preview to {out_path}")


def main() -> None:
    # Paths to resources (adjust if necessary)
    # ------------------------------------------------------------------
    # Filenames
    #
    # By default the script looks for ``map.dat`` (the world and chunk
    # definitions), ``tiles.png`` (the ground tileset), and
    # ``objects.png`` (the objects/decorations tileset).  If your files
    # have different names—for example, if you extracted tiles from
    # ``gdata02.vdf`` into a different PNG, or if your object sheet is
    # stored under another name—adjust these variables accordingly.  If
    # the files are missing the script will exit with an error.
    map_path = Path("map.dat")
    tiles_path = Path("tiles.png")
    objects_path = Path("objects.png")

    if not map_path.exists():
        raise SystemExit("map.dat not found in current directory")
    if not tiles_path.exists() or not objects_path.exists():
        raise SystemExit(
            "tiles.png and objects.png are required in the current directory"
        )

    # Load resources
    ground_tiles = load_tile_sheet(tiles_path)
    object_tiles = load_tile_sheet(objects_path)
    layers, chunk_map = read_map(map_path)

    # Render a 4×4 chunk preview of the top‑left area
    render_preview(
        layers,
        chunk_map,
        ground_tiles,
        object_tiles,
        Path("preview.png"),
        chunks_w=4,
        chunks_h=4,
    )


if __name__ == "__main__":
    main()
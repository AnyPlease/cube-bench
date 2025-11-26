from __future__ import annotations

"""virtual_cube.py
A lean, test-friendly utility for simulating and visualising a 3x3 Rubik's Cube.
The public surface is intentionally small:

    >>> cube = VirtualCube()
    >>> cube.scramble(20)
    >>> cube.to_image("scrambled.png")

Everything else is considered an implementation detail and may change.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import kociemba
from functools import lru_cache
from kociemba.pykociemba.facecube import FaceCube
import re
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
import pycuber as pc  # type: ignore – external dependency
from pycuber.solver import CFOPSolver
from PIL import Image, ImageDraw, ImageEnhance
import io, base64
import random
# import cube_bench.optimal.solver as sv

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    """Maps logical cube colours to RGB-255 triples."""

    colour_to_rgb: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "white": (255, 255, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 128, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
        }
    )

    def __getitem__(self, colour: str) -> Tuple[int, int, int]:
        return self.colour_to_rgb[colour.lower()]

@dataclass(frozen=True)
class NetLayout:
    """Pre-computed positions (top-left y, x) for each cube face in the 2-D net."""

    face_px: int  # width/height of one 3×3 face in pixels
    face_gap: int

    def __post_init__(self):
        object.__setattr__(self, "positions", self._compute_positions())

    def _compute_positions(self) -> Dict[str, Tuple[int, int]]:
        s, g = self.face_px, self.face_gap  # shorthand
        return {
            "U": (0, s + g),
            "L": (s + g, 0),
            "F": (s + g, s + g),
            "R": (s + g, 2 * (s + g)),
            "B": (s + g, 3 * (s + g)),
            "D": (2 * (s + g), s + g),
        }

    def canvas_shape(self) -> Tuple[int, int]:
        """Height, width of the full unfolded cube canvas in pixels."""
        h = 3 * self.face_px + 2 * self.face_gap
        w = 4 * self.face_px + 3 * self.face_gap
        return h, w
    

# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
@lru_cache(maxsize=262144)
def _co_eo_from_facelets(facelets: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Convert a Kociemba facelet string (URFDLB order, 54 chars) to
    (corner twists, edge flips). co in {0,1,2} (len=8), eo in {0,1} (len=12).
    """
    cc = FaceCube(facelets).toCubieCube()
    # cc.co and cc.eo are Python lists/arrays in the reference implementation
    return tuple(int(x) for x in cc.co[:8]), tuple(int(x) for x in cc.eo[:12])

class VirtualCube:
    """A lightweight facade around *pycuber*'s `Cube` with rendering helpers."""

    #: Human‑friendly labels for annotation – edit freely.
    FACE_LABELS = {"U": "Top", "L": "Left", "F": "Front", "R": "Right", "B": "Back", "D": "Down"}

    #: Faces in the order they appear in the unfolded net
    FACE_ORDER = ["U", "L", "F", "R", "B", "D"]

    #: Basic and double turns the cube understands (no slice/wide moves)
    AVAILABLE_MOVES: Tuple[str, ...] = (
        "R", "L", "U", "D", "F", "B",
        "R'", "L'", "U'", "D'", "F'", "B'",
        "R2", "L2", "U2", "D2", "F2", "B2",
    )

    _COLOR_ALIASES = {
        "w": "white", "white": "white",
        "y": "yellow","yellow":"yellow",
        "o": "orange","orange":"orange",
        "r": "red",   "red":   "red",
        "g": "green", "green": "green",
        "b": "blue",  "blue":  "blue",
    }

    def _canon(self, name: str) -> str:
        return self._COLOR_ALIASES.get(str(name).lower().strip(), str(name).lower().strip())

    # ---------------------------------------------------------------------
    # Construction & simple helpers
    # ---------------------------------------------------------------------

    def __init__(self) -> None:
        self._cube: pc.Cube = pc.Cube()
        self._palette = Palette()
    
    def __str__(self) -> str:
        return self._cube.__str__()
    
    def is_solved(self) -> bool:
        """True if each face is uniform (scheme-agnostic)."""
        
        def center_matching():
            for f in "ULFRBD":
                face = self._cube.get_face(f)
                c0 = face[1][1].colour
                if any(sq.colour != c0 for row in face for sq in row):
                    return False
            
            return True
        
        def lazy_matching():
            """Fallback matching"""
            solved_cube = pc.Cube()
            return self._cube.__str__() == solved_cube.__str__()

        return center_matching() or lazy_matching()
    
    def clone(self) -> "VirtualCube":
        """Return an independent copy of this VirtualCube."""
        new_cube = VirtualCube()
        new_cube._cube = self._cube.copy()  # pycuber’s safe copy
        return new_cube
    
    def get_distance(self) -> int:
        if self.is_solved():
            return 0

        s54 = self.to_kociemba()
        solution = sv.solve(s54)
        solution = solution.split(" ")

        distance = solution[-1]
        distance = re.search(r"\d+", distance)

        return int(distance.group())
    
    def corner_orientations(self) -> list[int]:
        co, _ = _co_eo_from_facelets(self.to_kociemba())
        return list(co)

    def edge_orientations(self) -> list[int]:
        _, eo = _co_eo_from_facelets(self.to_kociemba())
        return list(eo)

    def scramble(self, random_seed: int = 69, n_moves: int = 20, max_tries: int = 50) -> pc.Formula:
        """
        Apply n_moves random turns and return the scramble string.
        - Uses a local RNG (no global seeding side-effects).
        - Retries if the sequence accidentally solves the cube (capped by max_tries).
        - Avoids immediate same-face repeats to reduce trivial cancellations.
        """
        rng = np.random.default_rng(random_seed)

        # If you have faces like U, U', U2, map each move to its face letter
        def face_of(move: str) -> str:
            # Works for "U", "U'", "U2", "Rw", etc. Adjust if your notation differs.
            return move[0]

        # Simple “no same-face twice in a row” sampler
        def sample_moves() -> list[str]:
            seq = []
            last_face = None
            for _ in range(n_moves):
                candidates = [m for m in self.AVAILABLE_MOVES if face_of(m) != last_face] \
                            if last_face else list(self.AVAILABLE_MOVES)
                m = candidates[rng.integers(len(candidates))]
                seq.append(m)
                last_face = face_of(m)
            return seq

        # Try up to max_tries to avoid the (rare) identity/cancel-out scramble
        for _ in range(max_tries):
            moves = sample_moves()
            formula = pc.Formula(moves)
            self._cube(formula)
            if not self.is_solved():
                self.formula = formula
                # If you truly need a string, return str(...). If Formula is desired, change annotation.
                return self.formula.copy()
            # undo and try again
            self._cube(formula.reverse())

        # If we somehow failed all tries, just keep the last one anyway
        self.formula = formula
        return self.formula.copy() # pc.Formula

    def apply(self, moves: str) -> None:
        """Apply a move sequence given in standard notation (e.g. "R U R' U'")."""
        self._cube(moves)
    
    def solve(self):
        if self.is_solved():
            return ""

        s54 = self.to_kociemba()
        solution = sv.solve(s54)
        solution = solution.split(" ")

        for i, op in enumerate(solution):
            if op.endswith("1"):
                solution[i] = op[0]

            elif op.endswith("3"): 
                solution[i] = op[0] + "'"

        optimal_solution = " ".join(solution[:-1])

        return optimal_solution

    def front_face(self) -> List[List[str]]:
        """Return the current colours of the *Front* face (3x3 list)."""
        face = self._cube.get_face("F")
        return [[sq.colour for sq in row] for row in face]
    
    def reset(self):
        """Resets the cube back to solved state."""
        self._cube: pc.Cube = pc.Cube()
    
    def generate_sample(self, n_moves, idx: int):
        scramble = self.scramble(random_seed=idx, n_moves=n_moves)
        correct_move_str = str(scramble.reverse())

        # Generate 3 unique distractor move sequences (not equal to the correct solution)
        distractors = set()
        while len(distractors) < 3:
            moves = np.random.choice(self.AVAILABLE_MOVES, n_moves)
            move_seq = " ".join(moves)
            if move_seq != correct_move_str:
                distractors.add(move_seq)
        distractors = list(distractors)

        # Create the final options list and shuffle it.
        options = distractors + [correct_move_str]
        random.shuffle(options)

        # Create the dictionary and find the correct letter.
        options_dict = dict(zip("ABCD", options))
        correct_option = next(letter for letter, move in options_dict.items() if move == correct_move_str)

        sample = {
            "id": idx,
            "scramble_cube": self._cube.__str__(),
            "options": options_dict,
            "correct_option": correct_option,
            "image": self.to_image(),
            "difficulty": "easy" if n_moves == 1 else "medium" if n_moves == 2 else "hard",
            "front_face_colors": self.front_face(),  # for ReconstructionTest
        }

        self.reset()

        return sample
    
    def to_kociemba(self, net: str | None = None) -> str:
        """
        Export the cube to a 54-character Kociemba facelet string in URFDLB order,
        robust to any isomorphic recolor. We map *sticker colors -> face letters*
        using the current six center stickers as the ground truth.

        If `net` is provided (pycuber-style ASCII), we parse it; otherwise we read
        the cube object directly.
        """
        # Build color->face map from *current* centers (scheme-agnostic)
        color_to_face = {
            str(self._cube.get_face(f)[1][1].colour).lower(): f
            for f in "URFDLB"
        }
        if len(color_to_face) != 6:
            raise ValueError("Center colors must be unique; current scheme appears invalid.")

        def _token_to_faceletter(tok: str) -> str:
            # tok like 'y','o','g','w','r','b' or full names; normalize to full color
            col = self._canon(tok)  # 'y'->'yellow', etc., or lowercase passthrough
            try:
                return color_to_face[col]
            except KeyError as e:
                raise ValueError(f"Unknown sticker color token '{tok}' (-> '{col}') for current centers.") from e

        out: list[str] = []

        if net is None:
            # Read stickers directly from the cube in URFDLB, row-major (0..2, 0..2)
            for f in "URFDLB":
                face = self._cube.get_face(f)
                for r in range(3):
                    for c in range(3):
                        col = str(face[r][c].colour).lower()
                        try:
                            out.append(color_to_face[col])
                        except KeyError as e:
                            raise ValueError(f"Sticker color '{col}' not present in center mapping.") from e
        else:
            # Parse pycuber's ASCII net layout and then map tokens via centers
            rows = net.strip().splitlines()
            token_re = re.compile(r"\[([a-zA-Z]+)\]")  # accepts 'y' or 'yellow'

            faces_tokens: Dict[str, List[str]] = {k: [] for k in "ULFRBD"}
            for row_idx, row in enumerate(rows):
                tokens = token_re.findall(row)
                if not tokens:
                    continue
                if row_idx <= 2:  # top 3 rows => U
                    faces_tokens["U"].extend(tokens)
                elif 3 <= row_idx <= 5:  # middle band L F R B
                    if len(tokens) >= 12:
                        faces_tokens["L"].extend(tokens[0:3])
                        faces_tokens["F"].extend(tokens[3:6])
                        faces_tokens["R"].extend(tokens[6:9])
                        faces_tokens["B"].extend(tokens[9:12])
                else:  # bottom 3 rows => D
                    faces_tokens["D"].extend(tokens)

            # Convert tokens to face letters in Kociemba order URFDLB
            for f in "URFDLB":
                toks = faces_tokens[f]
                if len(toks) != 9:
                    raise ValueError(f"Face '{f}' does not have 9 tokens in provided net.")
                out.extend(_token_to_faceletter(t) for t in toks)

        assert len(out) == 54, f"Expected 54 facelets, got {len(out)}."
        return "".join(out)

    def from_kociemba(self, state54: str | None = None) -> str:
        """
        Construct the ASCII net for a Kociemba-ordered 54-char string.
        We:
        1) Use Kociemba to get a solution for `state54`,
        2) Recreate that state on a fresh cube,
        3) Recolor that fresh cube to match *this instance's current center scheme*,
        4) Return the ASCII net of that recolored cube.

        This keeps the output visually consistent with any prior isomorphic recolor.
        """
        if not state54:
            state54 = self.to_kociemba()  # derive from current cube

        # 1) get solution moves (state -> solved)
        moves = kociemba.solve(state54)

        # 2) recreate the state on a fresh, default-scheme cube
        temp = pc.Cube()
        temp(pc.Formula(moves).reverse())  # solved -> state

        # 3) recolor `temp` to match THIS cube's current center scheme
        #    Build a mapping: default_center_color -> current_center_color (by face)
        default_center_by_face = {
            f: str(pc.Cube().get_face(f)[1][1].colour).lower()  # defaults
            for f in "URFDLB"
        }
        current_center_by_face = {
            f: str(self._cube.get_face(f)[1][1].colour).lower()
            for f in "URFDLB"
        }
        recolor_map = {
            default_center_by_face[f]: current_center_by_face[f]
            for f in "URFDLB"
        }
        # sanity: we expect 6 unique target colors under a proper isomorphism
        if len(set(recolor_map.values())) != 6:
            # Not fatal for rendering, but indicates a non-bijective recolor.
            # We still proceed; you may want to enforce bijection in recolor_isomorphic.
            pass

        # Apply recolor on `temp` (assign lowercase strings)
        for f in "ULFRBD":
            face = temp.get_face(f)
            for r in range(3):
                for c in range(3):
                    sq = face[r][c]
                    src = str(sq.colour).lower()
                    tgt = recolor_map.get(src, src)  # default passthrough
                    sq.colour = tgt

        # Reindex internals so hashes reflect new colors if moves are applied later
        temp = temp.copy()

        # 4) return ASCII net in the *current* color scheme
        return temp.__str__()

    # ---- PUBLIC RENDER API -------------------------------------------------
    def render(self, *, cell_size: int = 60, sticker_border: int = 2, face_gap: int = 40, 
               return_type: str = "pil", file_path: Optional[Union[str, Path]] = None, dpi: int = 100, add_labels: bool = True):
        """
        Render the cube net and return it in a HuggingFace-friendly format.

        Parameters
        ----------
        return_type:
            One of:
              - "pil": return a PIL.Image (RGB)
              - "numpy": return a uint8 array (H,W,3)
              - "tensor": return a float tensor (3,H,W) in [0,1]
              - "bytes": PNG-encoded bytes
              - "base64": base64-encoded PNG string (UTF-8)
              - "figure": matplotlib Figure object (no saving)
              - "path": write file (requires file_path) and return Path
        file_path:
            If provided and return_type == "path" (or any type), the image is saved (PNG by default if suffix omitted).
        add_labels: Include face labels on the image.
        """
        canvas, layout = self._build_canvas(
            cell_size=cell_size,
            sticker_border=sticker_border,
            face_gap=face_gap,
        )

        # Build matplotlib figure only if we need labels OR figure/path
        need_mpl = add_labels or return_type in {"figure", "path"} or file_path is not None
        if need_mpl:
            fig = self._canvas_to_figure(canvas, layout, dpi=dpi, add_labels=add_labels)
            if file_path:
                file_path = Path(file_path)
                # Use suffix if given else default to .png
                if not file_path.suffix:
                    file_path = file_path.with_suffix(".png")
                fig.savefig(
                    file_path,
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.1,
                    facecolor=fig.get_facecolor(),
                )
            # Extract numpy array (with labels baked in) if needed in another format
            if return_type not in {"figure", "path"}:
                fig.canvas.draw()
                # New API: get an RGBA buffer (H * W * 4 bytes)
                rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()

                # Detect and handle HiDPI/Retina scaling (buffer size > logical size)
                scale = int(np.sqrt(rgba.size / (w * h * 4)))
                if scale > 1:
                    w, h = w * scale, h * scale

                rgba = rgba.reshape(h, w, 4)

                # Drop alpha (matplotlib may have composited background already)
                canvas = rgba[..., :3].copy()   # copy if you plan to close fig soon
                
            if return_type != "figure":
                plt.close(fig)
        else:
            fig = None  # not built

        if return_type == "figure":
            return fig
        
        if return_type == "path":
            if not file_path:
                raise ValueError("file_path must be provided when return_type='path'.")
            return Path(file_path)
        
        if return_type == "numpy":
            return canvas  # (H,W,3) uint8
        
        if return_type == "pil":
            if Image is None:
                raise RuntimeError("Pillow not installed; cannot return PIL image.")
            return Image.fromarray(canvas, mode="RGB")
        
        if return_type == "tensor":
            tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
            return tensor  # (3,H,W)
            
        if return_type in {"bytes", "base64"}:
            if Image is None:
                raise RuntimeError("Pillow not installed; cannot encode image.")
            pil_img = Image.fromarray(canvas, mode="RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data = buf.getvalue()
            if return_type == "bytes":
                return data
            else:
                return base64.b64encode(data).decode("utf-8")
        raise ValueError(f"Unknown return_type '{return_type}'.")

    # Backwards-compatible alias
    def to_image(self, file_path: Optional[Union[str, Path]] = None, **kwargs):
        """
        Legacy wrapper: if file_path supplied -> returns Path else returns PIL image.
        """
        return_type = "path" if file_path else "pil"
        return self.render(file_path=file_path, return_type=return_type, **kwargs)
    
    # ---- IMAGE VARIANTS / AUGMENTATIONS ------------------------------------
    def _augment_image(self, img: Image.Image, variant: str, *, brightness: float = 0.8) -> Image.Image:
        """
        Stateless image-only transforms applied to a rendered cube image.

        Supported variants:
            - "clean"   : return as-is
            - "rot90"   : rotate image by +90° (expand canvas)
            - "occl"    : central horizontal black band (~15% height)
            - "bright"  : brightness jitter (default 0.8x, darker)

        Notes:
            - 'recolor' is NOT here; it's a state-level variant (see render_variant).
        """
        variant = (variant or "clean").lower()
        if variant == "clean":
            return img

        if variant == "occl":
            out = img.copy()
            draw = ImageDraw.Draw(out)
            w, h = out.size
            band_h = max(6, int(0.15 * h))
            top = (h - band_h) // 2
            draw.rectangle([0, top, w, top + band_h], fill=(0, 0, 0))
            return out

        if variant == "bright":
            enh = ImageEnhance.Brightness(img)
            return enh.enhance(brightness)

        raise ValueError(f"Unknown image variant '{variant}'. Supported: clean, rot90, occl, bright.")


    def recolor_isomorphic(self, color_map: Dict[str, str]) -> None:
        """
        Mutates sticker colors safely, then rebuilds pycuber internals so moves still work.
        """
        # Build color->center mapping (whatever types pycuber uses)
        centers = { str(self._cube.get_face(f)[1][1].colour).lower(): self._cube.get_face(f)[1][1].colour
                    for f in "URFDLB" }

        def _canon(name: str) -> str:
            return self._COLOR_ALIASES.get(str(name).lower().strip(), str(name).lower().strip())

        norm = { _canon(k): _canon(v) for k, v in color_map.items() }
        # Validate targets exist among centers (same scheme)
        missing = set(norm.values()) - set(centers.keys())
        if missing:
            raise ValueError(f"Unknown target colors in this cube scheme: {sorted(missing)}")

        # Recolor each sticker to the canonical center value
        for f in self.FACE_ORDER:
            face = self._cube.get_face(f)
            for r in range(3):
                for c in range(3):
                    sq = face[r][c]
                    src = str(sq.colour).lower()
                    if src in norm:
                        sq.colour = centers[norm[src]]

        # 🔧 Reindex internal containers so hashes match new colours
        # Easiest: force a full copy which rebuilds sets/dicts with current hashes
        self._cube = self._cube.copy()
    
    def render_variant(self, variant: str, *, cell_size: int = 60, sticker_border: int = 2, face_gap: int = 40, dpi: int = 100, add_labels: bool = True, recolor_map: dict[str, str] | None = None, return_type: str = "pil", file_path: Optional[Union[str, Path]] = None):
        """
        Render a specific variant of the current cube.

        Variants
        --------
        - "clean" : normal render
        - "recolor" : clone cube, apply isomorphic recolor (requires recolor_map), then render
        - "occl"  : image-only occlusion band
        - "bright": image-only brightness jitter (0.8x)

        Notes
        -----
        - 'recolor' is a STATE variant (uses a cloned, recolored cube and re-renders).
        - All others are IMAGE variants (post-process the rendered image).
        - `return_type` can be "pil" | "numpy" | "tensor" | "bytes" | "base64" | "path" | "figure".
        """
        v = (variant or "clean").lower()

        if v == "recolor":
            if not recolor_map:
                raise ValueError("render_variant('recolor') requires `recolor_map`.")
            c = self.clone()
            c.recolor_isomorphic(recolor_map)
            # Always render to PIL first, then convert if needed
            pil_img = c.render(
                cell_size=cell_size,
                sticker_border=sticker_border,
                face_gap=face_gap,
                dpi=dpi,
                add_labels=add_labels,
                return_type="pil",
            )
            # Convert to the requested return_type
            if return_type == "pil":
                return pil_img
            elif return_type == "numpy":
                return np.asarray(pil_img, dtype=np.uint8)
            elif return_type == "tensor":
                return torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).float() / 255.0
            elif return_type in {"bytes", "base64", "path", "figure"}:
                # reuse the standard pipeline by re-encoding
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                data = buf.getvalue()
                if return_type == "bytes":
                    return data
                if return_type == "base64":
                    return base64.b64encode(data).decode("utf-8")
                if return_type == "path":
                    if not file_path:
                        raise ValueError("file_path must be provided when return_type='path'.")
                    file_path = Path(file_path)
                    if not file_path.suffix:
                        file_path = file_path.with_suffix(".png")
                    with open(file_path, "wb") as f:
                        f.write(data)
                    return file_path
                if return_type == "figure":
                    # Minimal support: embed as MPL figure background
                    fig, ax = plt.subplots(figsize=(pil_img.width / dpi, pil_img.height / dpi), dpi=dpi)
                    ax.imshow(pil_img)
                    ax.axis("off")
                    fig.tight_layout(pad=0)
                    return fig
            else:
                raise ValueError(f"Unknown return_type '{return_type}'.")

        # IMAGE-only variants (clean/occl/bright)
        pil_img = self.render(
            cell_size=cell_size,
            sticker_border=sticker_border,
            face_gap=face_gap,
            dpi=dpi,
            add_labels=add_labels,
            return_type="pil",
        )
        pil_img = self._augment_image(pil_img, v)

        if return_type == "pil":
            return pil_img
        if return_type == "numpy":
            return np.asarray(pil_img, dtype=np.uint8)
        if return_type == "tensor":
            return torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).float() / 255.0
        if return_type in {"bytes", "base64"}:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data = buf.getvalue()
            return data if return_type == "bytes" else base64.b64encode(data).decode("utf-8")
        if return_type == "path":
            if not file_path:
                raise ValueError("file_path must be provided when return_type='path'.")
            file_path = Path(file_path)
            if not file_path.suffix:
                file_path = file_path.with_suffix(".png")
            pil_img.save(file_path, format="PNG")
            return file_path
        if return_type == "figure":
            fig, ax = plt.subplots(figsize=(pil_img.width / dpi, pil_img.height / dpi), dpi=dpi)
            ax.imshow(pil_img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            return fig

        raise ValueError(f"Unknown return_type '{return_type}'.")


    def to_image_variant(
        self,
        variant: str,
        *,
        recolor_map: dict[str, str] | None = None,
        **kwargs
    ) -> Image.Image:
        """
        Convenience wrapper: always returns a PIL image for a variant.
        """
        return self.render_variant(
            variant,
            recolor_map=recolor_map,
            return_type="pil",
            **kwargs
        )


    def render_variants(
        self,
        variants: list[str],
        *,
        recolor_map: dict[str, str] | None = None,
        return_type: str = "pil",
        **kwargs
    ) -> dict[str, Image.Image | np.ndarray | torch.Tensor | bytes | str | Path]:
        """
        Batch-render multiple variants. Returns a dict {variant_name: image_like}.
        """
        out = {}
        for v in variants:
            if v.lower() == "recolor":
                out[v] = self.render_variant(v, recolor_map=recolor_map, return_type=return_type, **kwargs)
            else:
                out[v] = self.render_variant(v, return_type=return_type, **kwargs)
        return out



    # ---- INTERNAL: make raw canvas ----------------------------------------
    def _build_canvas(self, *, cell_size: int, sticker_border: int, face_gap: int):
        layout = NetLayout(face_px=3 * cell_size, face_gap=face_gap)
        h, w = layout.canvas_shape()
        canvas = np.full((h, w, 3), 127, dtype=np.uint8)
        for face_key in self.FACE_ORDER:
            self._paint_face(
                canvas,
                face_key,
                origin=layout.positions[face_key],
                cell_size=cell_size,
                sticker_border=sticker_border,
            )
        return canvas, layout
    
    def _canvas_to_figure(self, canvas: np.ndarray, layout: NetLayout, *, dpi: int, add_labels: bool):
        canvas_h, canvas_w = canvas.shape[:2]
        fig_w_in, fig_h_in = canvas_w / dpi, canvas_h / dpi
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        grey = (125 / 255,) * 3
        fig.patch.set_facecolor(grey)
        ax.set_position([0, 0, 1, 1])
        ax.imshow(canvas, interpolation="nearest")
        ax.axis("off")
        if add_labels:
            for face_key in self.FACE_ORDER:
                y0, x0 = layout.positions[face_key]
                ax.text(
                    x0 + layout.face_px / 2,
                    y0 - 6,
                    self.FACE_LABELS[face_key],
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, linewidth=0),
                )
        fig.tight_layout(pad=0)
        return fig

    def _paint_face(self, canvas: np.ndarray, face_key: str, *, origin: Tuple[int, int], cell_size: int, sticker_border: int,) -> None:
        """Blit a single 3x3 face onto the *canvas* at *origin*."""
        y0, x0 = origin
        face_px = 3 * cell_size
        face_img = np.zeros((face_px, face_px, 3), dtype=np.uint8)  # black background for borders

        face_grid = self._cube.get_face(face_key)
        for r in range(3):
            for c in range(3):
                rgb = self._palette[str(face_grid[r][c].colour)]
                rs, re = r * cell_size + sticker_border, (r + 1) * cell_size - sticker_border
                cs, ce = c * cell_size + sticker_border, (c + 1) * cell_size - sticker_border
                face_img[rs:re, cs:ce] = rgb

        canvas[y0 : y0 + face_px, x0 : x0 + face_px] = face_img


if __name__ == "__main__":
    cube = VirtualCube()
    cube.scramble(n_moves=10)
    cube.to_image("cube_scramble.png")

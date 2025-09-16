"""Lightweight wrapper for the Hugging Face Myna checkpoints."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel

from marble.core.base_encoder import BaseEncoder


def _resolve_model_name(name: str, aliases: Dict[str, str]) -> str:
    """Resolve short aliases like ``"myna-hybrid"`` to full repo IDs."""

    return aliases.get(name.lower(), name)


def _as_pair(value: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("Expected a 2-tuple for patch size or spec size")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


class MynaEncoder(BaseEncoder):
    """Thin wrapper around :func:`transformers.AutoModel.from_pretrained`.

    The encoder accepts mel spectrogram inputs with shape ``[B, 1, H, W]`` (or the
    unbatched variants documented in the Hugging Face README) and emits chunked
    embeddings stacked along a time axis. Long spectrograms are split into
    non-overlapping windows whose widths respect the model's positional
    embedding limit. Each window is evaluated in a single batch, so the final
    tensor has shape ``[B, 1, num_chunks, hidden_dim]`` without any averaging of
    embeddings.
    """

    MODEL_ALIASES: Dict[str, str] = {
        "myna-hybrid": "oriyonay/myna-hybrid",
        "myna-base": "oriyonay/myna-base",
        "myna-vertical": "oriyonay/myna-vertical",
        "myna-85m": "oriyonay/myna-85m",
    }

    MODEL_DIMENSIONS: Dict[str, int] = {
        "oriyonay/myna-hybrid": 768,
        "oriyonay/myna-base": 384,
        "oriyonay/myna-vertical": 384,
        "oriyonay/myna-85m": 1536,
    }

    def __init__(
        self,
        model_name: str = "oriyonay/myna-hybrid",
        *,
        pre_trained_folder: Optional[str] = None,
        train_mode: str = "freeze",
        trust_remote_code: bool = True,
        chunk_frames: Optional[int] = None,
    ) -> None:
        super().__init__()

        repo = pre_trained_folder or _resolve_model_name(model_name, self.MODEL_ALIASES)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=trust_remote_code)

        if train_mode not in {"freeze", "full"}:
            raise ValueError("train_mode must be either 'freeze' or 'full'.")

        requires_grad = train_mode == "full"
        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.model.train(mode=requires_grad)

        self.embedding_dim = self.MODEL_DIMENSIONS.get(repo, getattr(self.model.config, "dim", 0))

        spec_height, spec_width = _as_pair(getattr(self.model.config, "spec_size", (128, 96)))
        patch_height, patch_width = _as_pair(getattr(self.model.config, "patch_size", (16, 16)))
        additional_patch = getattr(self.model.config, "additional_patch_size", None)

        self.spec_height = int(spec_height)
        self.spec_width = int(spec_width)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)

        multiples = [self.patch_width]
        if additional_patch is not None:
            _, add_width = _as_pair(additional_patch)
            multiples.append(int(add_width))

        required_multiple = multiples[0]
        for value in multiples[1:]:
            required_multiple = _lcm(required_multiple, value)
        self.required_multiple = required_multiple

        # Maximum number of mel frames handled in a single forward pass.
        max_frames = self.spec_width if self.spec_width > 0 else required_multiple
        if chunk_frames is not None:
            if chunk_frames <= 0:
                raise ValueError("chunk_frames must be a positive integer if provided")
            max_frames = min(max_frames, chunk_frames)

        # Ensure chunk width is a multiple of all required patch sizes.
        self.chunk_frames = max(self.required_multiple, max_frames - max_frames % self.required_multiple)

    def _standardise_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:  # [B, H, W]
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(
                "Expected input tensor with 2–4 dimensions corresponding to (H, W), "
                "(B, H, W) or (B, C, H, W), received shape "
                f"{tuple(x.shape)}."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "Myna expects a single spectrogram channel; received tensor with "
                f"{x.shape[1]} channels."
            )
        return x

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError("multiple must be positive")
        return max(multiple, ((value + multiple - 1) // multiple) * multiple)

    def _pad_last_dim(self, x: torch.Tensor, target: int) -> torch.Tensor:
        pad = target - x.size(-1)
        if pad <= 0:
            return x
        return F.pad(x, (0, pad))

    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        del input_len

        x = self._standardise_input(input_tensor)

        if x.size(2) != self.spec_height:
            if x.size(2) > self.spec_height:
                x = x[..., : self.spec_height, :]
            else:
                pad_h = self.spec_height - x.size(2)
                x = F.pad(x, (0, 0, 0, pad_h))

        step = self.chunk_frames if self.chunk_frames > 0 else x.size(-1)
        if step <= 0:
            raise RuntimeError("Invalid chunk size computed for Myna encoder")

        if x.size(-1) == 0:
            chunk_slices = [x[..., :0]]
        else:
            chunk_slices = [x[..., start : min(start + step, x.size(-1))] for start in range(0, x.size(-1), step)]

        chunk_tensors: list[torch.Tensor] = []
        chunk_widths: list[int] = []
        for chunk in chunk_slices:
            target = self._round_up(chunk.size(-1), self.required_multiple)
            chunk_tensors.append(self._pad_last_dim(chunk, target))
            chunk_widths.append(target)

        params = next(self.model.parameters())
        device = params.device
        dtype = params.dtype

        grouped_batches: list[torch.Tensor] = []
        group_sizes: list[int] = []
        current_group: list[torch.Tensor] = []
        current_width: Optional[int] = None

        for chunk_tensor, width in zip(chunk_tensors, chunk_widths):
            if current_width is None or width == current_width:
                current_group.append(chunk_tensor)
                current_width = width
            else:
                grouped_batches.append(torch.cat(current_group, dim=0))
                group_sizes.append(len(current_group))
                current_group = [chunk_tensor]
                current_width = width

        if current_group:
            grouped_batches.append(torch.cat(current_group, dim=0))
            group_sizes.append(len(current_group))

        chunk_outputs: list[torch.Tensor] = []
        for batch_tensor, count in zip(grouped_batches, group_sizes):
            batch_tensor = batch_tensor.to(device=device, dtype=dtype)
            outputs = self.model(batch_tensor)

            if not isinstance(outputs, torch.Tensor):
                raise TypeError(
                    "Unexpected output type from Myna model; expected torch.Tensor but "
                    f"received {type(outputs)!r}."
                )

            chunk_outputs.append(outputs.view(count, x.size(0), -1))

        if not chunk_outputs:
            empty = torch.empty(x.size(0), 0, self.embedding_dim, device=device, dtype=dtype)
            return empty.unsqueeze(1)

        embeddings = torch.cat(chunk_outputs, dim=0).permute(1, 0, 2).contiguous()

        return embeddings.unsqueeze(1)

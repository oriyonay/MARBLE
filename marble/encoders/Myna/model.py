"""Lightweight wrapper for the Hugging Face Myna checkpoints."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import AutoModel

from marble.core.base_encoder import BaseEncoder


def _resolve_model_name(name: str, aliases: Dict[str, str]) -> str:
    """Resolve short aliases like ``"myna-hybrid"`` to full repo IDs."""

    return aliases.get(name.lower(), name)


class MynaEncoder(BaseEncoder):
    """Thin wrapper around :func:`transformers.AutoModel.from_pretrained`."""

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

        self.embedding_dim = self.MODEL_DIMENSIONS.get(repo, 0)

    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        del input_len

        params = next(self.model.parameters())
        device = params.device
        dtype = params.dtype

        outputs = self.model(input_tensor.to(device=device, dtype=dtype))

        if isinstance(outputs, torch.Tensor):
            embeddings = outputs
        else:
            # Some subclasses may wrap the tensor in a dataclass, but Myna does not
            raise TypeError(
                "Unexpected output type from Myna model; expected torch.Tensor but "
                f"received {type(outputs)!r}."
            )

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(1)

        return embeddings

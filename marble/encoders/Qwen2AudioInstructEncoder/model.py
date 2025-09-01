# marble/encoders/Qwen2AudioInstructEncoder/model.py
from typing import Dict, Optional

import torch
import torch.nn as nn
from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform
from .processing_qwen2_audio import Qwen2AudioProcessor
from .modeling_qwen2_audio import Qwen2AudioForConditionalGeneration


class Qwen2AudioInstructEncoder(BaseEncoder):
    """
    A wrapper around the Qwen2-Audio-7B-Instruct encoder with optional freezing or full fine-tuning.
    """
    NAME = "Qwen2AudioInstructEncoder"
    HUGGINGFACE_MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
    N_TRANSFORMER_LAYERS = 32
    SAMPLING_RATE = 16000
    TOKEN_RATE = 25
    NUM_FEATURES = 1280

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full"]
        attn_implementation: str = "sdpa"  # or "flash_attention_2"
    ) -> None:
        super().__init__()
        repo = pre_trained_folder or self.HUGGINGFACE_MODEL_NAME

        # Load processor (for feature_extractor)
        self.processor = Qwen2AudioProcessor.from_pretrained(
            repo,
        )
        self.feature_extractor = self.processor.feature_extractor
        self.sample_rate = self.feature_extractor.sampling_rate

        # Load encoder config and model
        print(f"Loading Qwen2AudioForConditionalGeneration from {repo}")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            repo,
            attn_implementation=attn_implementation,
        ).audio_tower

        # Configure training mode
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def forward(
        self,
        input_features: torch.FloatTensor,
        output_hidden_states: bool = True,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass through Qwen2AudioEncoder.

        Args:
            input_features (torch.FloatTensor): Log-mel features, shape (batch, n_mels, seq_len)
            attention_mask (torch.BoolTensor, optional): 1D feature mask, shape (batch, seq_len)
        Returns:
            Model outputs (e.g., last_hidden_state)
        """
        # Move inputs to model device
        device = next(self.model.parameters()).device
        input_features = input_features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = self.model(
            input_features=input_features,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        return outputs


class Qwen2AudioInstructFeatureExtractor(BaseAudioTransform):
    """
    Audio-to-feature transform using Qwen2AudioProcessor.
    """
    NAME = "Qwen2AudioInstructFeatureExtractor"
    HUGGINGFACE_MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
    N_TRANSFORMER_LAYERS = 32
    SAMPLING_RATE = 16000
    TOKEN_RATE = 25
    NUM_FEATURES = 1280

    def __init__(
        self,
        pre_trained_folder: str = None,
        squeeze: bool = True,  # 是否压缩为 1D
    ) -> None:
        super().__init__()
        repo = pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        self.processor = Qwen2AudioProcessor.from_pretrained(
            repo,
        )
        self.feature_extractor = self.processor.feature_extractor
        self.squeeze = squeeze

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sample["input_features"]: torch.Tensor (1D) or List[torch.Tensor] each 1D
            sample["sampling_rate"]: int
        Returns:
            sample with keys:
              - "input_features": torch.FloatTensor (batch, n_mels, seq_len)
        """
        x = sample["input_features"].squeeze() # waveform
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 1, "Input features must be a 1D tensor (batch_size=1)"
        # while it also supports list of ndarray, we disable it for now
        # this class should be used in the dataloader
        sr = sample.get("sampling_rate", self.feature_extractor.sampling_rate)

        feats = self.feature_extractor(
            x,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        x = feats["input_features"]
        if self.squeeze:
            x = x.squeeze(0)
        assert x.ndim == 2, f"Input features must be 2D tensor [num_mel_bins=128, seq_len], got {x.ndim}D, shape {x.shape}"
        sample["input_features"] = x  # [num_mel_bins=128, seq_len]
        return sample


if __name__ == "__main__":
    # 简单测试
    import torch

    local_repo = "Qwen/Qwen2-Audio-7B-Instruct"
    encoder = Qwen2AudioInstructEncoder(pre_trained_folder=local_repo)
    feature_extractor = Qwen2AudioInstructFeatureExtractor(pre_trained_folder=local_repo, squeeze=True)

    # 1. make a input
    import librosa
    audio_path = "/aifs4su/mmcode/codeclm/MARBLE2/data/GTZAN/genres/blues/blues.00000.wav"
    audio = librosa.load(audio_path, sr=16000)[0]
    

    # 2. 提取特征
    sample = {"input_features": torch.tensor(audio), "sampling_rate": 16000}
    sample = feature_extractor(sample)
    input_features = sample["input_features"].unsqueeze(0)  # 添加 batch 维度

    # 3. 前向计算
    outputs = encoder(input_features=input_features)
    # 打印 hidden_states 的数量
    print("Layer count:", len(outputs.hidden_states))

    # 打印 last_hidden_state 的 shape
    print("Last hidden state shape:", outputs.last_hidden_state.shape)

    # 计算并打印 last_hidden_state 的元素和
    total_sum = outputs.last_hidden_state.sum()
    print("Last hidden state sum:", total_sum.item())

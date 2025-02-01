from __future__ import annotations

import torch

from ..base import BaseModel
from .video_llama_prompt import VideoLlamaPromptMixin


class VideoLlama(VideoLlamaPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        self.device = 'cuda:0'

        from transformers import AutoProcessor, AutoModelForCausalLM

        assert model_path is not None
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation='flash_attention_2')
        self.model.cuda().eval()

        self.VIDEO_LLM = not model_path.lower().endswith('image')

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': {'image_path': s['value']}}
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': {'video_path': s['value']}}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        inputs = self.processor(conversation=messages, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self.model.generate(**inputs, **self.generate_kwargs)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

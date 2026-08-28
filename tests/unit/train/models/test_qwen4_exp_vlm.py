import pytest
import torch

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen4_exp import Qwen4ExpConfig, Qwen4ExpForCausalLM, Qwen4ExpVLMConfig
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def _config():
    text_config = Qwen4ExpConfig(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["linear_attention"],
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        hc_count=4,
        hc_lowrank=64,
        ple_layer_ids=[],
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=2,
        use_grouped_mm=False,
    )
    text_config._attn_implementation = "flash_attention_3"
    return Qwen4ExpVLMConfig(
        text_config=text_config.to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 4,
            "out_hidden_size": 256,
        },
        image_token_id=250,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )


def test_image_forward_backward():
    config = _config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen4ExpForCausalLM(config)
    inject_prime_lm_head(model)

    vision_config = config.vision_config
    patch_dim = (
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * vision_config.patch_size
        * vision_config.patch_size
    )
    image_grid_thw = torch.tensor([[1, 2, 2]], device="cuda")
    pixel_values = torch.randn(4, patch_dim, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.tensor([[11, 12, config.image_token_id, 13, 14]], device="cuda")
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == config.image_token_id] = 1

    output = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        seq_lens=torch.tensor([input_ids.shape[1]], device="cuda"),
    )
    output["logits"].sum().backward()

    assert output["logits"].shape == (*input_ids.shape, config.text_config.vocab_size)
    assert model.model.language_model.embed_tokens.weight.grad is not None
    assert model.model.visual.patch_embed.proj.weight.grad is not None

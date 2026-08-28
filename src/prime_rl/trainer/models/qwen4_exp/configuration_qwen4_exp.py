from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig

SPARSE_LAYER_TYPE = "qwen_sparse_attention"
QSA_FIELDS = ("indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio")


class Qwen4ExpVisionConfig(Qwen3_5MoeVisionConfig):
    model_type = "qwen4_exp_vision"

    def __init__(self, out_hidden_size=2560, **kwargs):
        super().__init__(out_hidden_size=out_hidden_size, **kwargs)


class Qwen4ExpConfig(PretrainedConfig):
    model_type = "qwen4_exp_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2560,
        num_hidden_layers=48,
        num_attention_heads=24,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        hc_count=4,
        hc_lowrank=320,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        output_gate_type="sigmoid",
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        ple_layer_ids=None,
        ple_embed_dim=None,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        seed=1234,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        load_balance_coeff=None,
        use_grouped_mm=True,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.pad_token_id = pad_token_id

        kwargs.setdefault("partial_rotary_factor", 0.25)

        if layer_types is None:
            interval = kwargs.pop("full_attention_interval", 4)
            layer_types = [
                "linear_attention" if (i + 1) % interval else "full_attention" for i in range(self.num_hidden_layers)
            ]
        self.layer_types = ["full_attention" if t == SPARSE_LAYER_TYPE else t for t in layer_types]
        unsupported = sorted(set(self.layer_types) - {"linear_attention", "full_attention"})
        if unsupported:
            raise ValueError(f"unsupported Qwen4-Exp layer types: {unsupported}")

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.output_gate_type = output_gate_type

        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio

        self.ple_layer_ids = sorted(set(ple_layer_ids)) if ple_layer_ids else []
        self.ple_embed_dim = hidden_size if ple_embed_dim is None else ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.split_ngram_parts = split_ngram_parts
        self.seed = 1234 if seed is None else seed

        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm

        self._validate_architecture()
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_architecture(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types must contain one entry per hidden layer; got {len(self.layer_types)} for "
                f"{self.num_hidden_layers} layers"
            )
        out_of_range = [i for i in self.ple_layer_ids if not 1 <= i <= len(self.layer_types)]
        if out_of_range:
            raise ValueError(
                f"ple_layer_ids must be one-indexed layer numbers in [1, {len(self.layer_types)}]; got {out_of_range}"
            )
        on_sparse = [i for i in self.ple_layer_ids if self.layer_types[i - 1] != "linear_attention"]
        if on_sparse:
            raise ValueError(f"Qwen4-Exp per-layer embeddings only run on linear_attention layers; got {on_sparse}")
        if self.indexer_compress_ratio <= 0:
            raise ValueError(f"indexer_compress_ratio must be positive; got {self.indexer_compress_ratio}")
        if self.indexer_budget <= 0:
            raise ValueError(f"indexer_budget must be positive; got {self.indexer_budget}")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError(
                f"indexer_budget ({self.indexer_budget}) must be divisible by "
                f"indexer_compress_ratio ({self.indexer_compress_ratio})"
            )
        if self.hc_count < 2:
            raise ValueError(f"Qwen4-Exp hyper-connections need hc_count > 1; got {self.hc_count}")
        if self.output_gate_type not in ("sigmoid", "silu"):
            raise ValueError(f"unsupported output gate activation: {self.output_gate_type}")
        missing = [f for f in QSA_FIELDS if getattr(self, f) is None]
        if missing:
            raise ValueError(f"Qwen4-Exp sparse attention is missing required fields: {missing}")
        if self.indexer_kv_heads != 1:
            raise ValueError(f"Qwen4-Exp sparse attention requires indexer_kv_heads=1; got {self.indexer_kv_heads}")
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads <= 0 or self.ple_embed_dim % ngram_heads:
            raise ValueError(
                f"ple_embed_dim ({self.ple_embed_dim}) must be divisible by the number of n-gram heads ({ngram_heads})"
            )

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def dense_equivalent_seqlen(self) -> int:
        return (self.block_topk + 1) * self.indexer_compress_ratio - 1


def _as_config(config_cls: type[PretrainedConfig], value) -> PretrainedConfig:
    if value is None:
        return config_cls()
    return config_cls(**value) if isinstance(value, dict) else value


class Qwen4ExpVLMConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    sub_configs = {"vision_config": Qwen4ExpVisionConfig, "text_config": Qwen4ExpConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.text_config = _as_config(Qwen4ExpConfig, text_config)
        if isinstance(vision_config, dict) and vision_config.get("model_type") == "qwen4_exp":
            vision_config = {**vision_config, "model_type": Qwen4ExpVisionConfig.model_type}
        self.vision_config = _as_config(Qwen4ExpVisionConfig, vision_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

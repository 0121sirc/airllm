from .airllm_base import AirLLMBaseModel


class AirLLMQwen3(AirLLMBaseModel):
    """Qwen3 family (dense ``Qwen3ForCausalLM`` and MoE ``Qwen3MoeForCausalLM``).

    The MoE variant stores one ``Qwen3MoeMLP`` per expert under ``mlp.experts`` and only executes
    the experts a token routes to (top-k). Like Kimi K3, we stream those experts individually so a
    layer costs a handful of small tensor reads instead of the whole shard (~1.1GB for Qwen3-30B).
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
            'expert_prefix': 'mlp.experts',
        }

    def get_use_better_transformer(self):
        return False
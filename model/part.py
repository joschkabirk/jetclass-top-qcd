import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "mod.cls_token",
        }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


# standard model configuration from
# https://github.com/jet-universe/particle_transformer/blob/main/networks/example_ParticleTransformer.py#L26-L44  # noqa: E501
part_default_kwargs = dict(
    input_dim=7,
    num_classes=10,
    # network configurations
    pair_input_dim=4,
    use_pre_activation_pair=False,
    embed_dims=[128, 512, 128],
    pair_embed_dims=[64, 64, 64],
    num_heads=8,
    num_layers=8,
    num_cls_layers=2,
    block_params=None,
    cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
    fc_params=[],
    activation="gelu",
    # misc
    trim=True,
    for_inference=False,
)

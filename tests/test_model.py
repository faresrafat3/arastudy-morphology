import torch

from src.models.transformer import AraStudyTransformer, count_parameters


def test_model_creation() -> None:
    model = AraStudyTransformer.from_config("configs/model/base_30m.yaml", vocab_size=16000)
    params = count_parameters(model)
    assert 25_000_000 < params["total"] < 32_000_000


def test_forward_pass() -> None:
    model = AraStudyTransformer.from_config("configs/model/base_30m.yaml", vocab_size=16000)
    x = torch.randint(0, 16000, (2, 128))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 128, 16000)
    assert loss is not None


def test_rope() -> None:
    model = AraStudyTransformer.from_config("configs/model/base_30m.yaml", vocab_size=16000)
    for name, _ in model.named_parameters():
        assert "pos" not in name.lower()


def test_weight_tying() -> None:
    model = AraStudyTransformer.from_config("configs/model/base_30m.yaml", vocab_size=16000)
    assert model.tok_embeddings.weight is model.output.weight

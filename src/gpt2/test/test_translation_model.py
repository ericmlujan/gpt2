from gpt2.models.translation_model import TransformerTranslationModel


class TestTransformerTranslationModel:
    def test_forward(self):
        model = TransformerTranslationModel()
        x = model.tokenizer.encode("My name is Eric!").unsqueeze(0)

        prev_output = model.tokenizer.encode("stub").unsqueeze(0)
        output = model.forward(x, prev_output)
        assert output.shape == (1, prev_output.shape[1], model.tokenizer.vocab_size())

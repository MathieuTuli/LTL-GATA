from pathlib import Path

from transformers import DistilBertModel, DistilBertTokenizer

PRETRAINED_LANGUAGE_MODEL = None
TOKENIZER = None


def get_model_tokenizer(model: str, checkpoint: Path = None):
    global PRETRAINED_LANGUAGE_MODEL
    global TOKENIZER
    if PRETRAINED_LANGUAGE_MODEL is None:
        if model == 'bert':
            if checkpoint is None:
                checkpoint = 'distilbert-base-uncased'
            PRETRAINED_LANGUAGE_MODEL = DistilBertModel.from_pretrained(
                checkpoint)
            for param in PRETRAINED_LANGUAGE_MODEL.parameters():
                param.requires_grad = False
            TOKENIZER = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased')

    return PRETRAINED_LANGUAGE_MODEL, TOKENIZER

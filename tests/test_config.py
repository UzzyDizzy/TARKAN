"""Paper §4.3 hyperparameters must match exactly, and label spaces must be correct."""
from config import (BIO_TAGS, CONFIG, NUM_BIO_TAGS, NUM_POLARITIES, POLARITIES)


def test_hyperparams_match_paper():
    assert CONFIG.hidden_dim == 768
    assert CONFIG.max_text_len == 128
    assert CONFIG.batch_size == 16
    assert CONFIG.learning_rate == 2e-5
    assert CONFIG.dropout == 0.3
    assert CONFIG.top_m_triples == 10
    assert CONFIG.lambda1 == 0.5 and CONFIG.lambda2 == 0.5
    assert CONFIG.bootstrap_samples == 1000


def test_label_spaces():
    assert NUM_BIO_TAGS == 7
    assert set(BIO_TAGS) == {"O", "B-POS", "I-POS", "B-NEU", "I-NEU", "B-NEG", "I-NEG"}
    assert NUM_POLARITIES == 3
    assert POLARITIES == ["POS", "NEU", "NEG"]


def test_model_ids():
    assert CONFIG.text_model_id == "vinai/bertweet-base"
    assert CONFIG.visual_model_id == "openai/clip-vit-base-patch32"

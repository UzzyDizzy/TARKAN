"""BIO<->span round-trip and joint reconstruction from per-aspect ($T$) records."""
from data import RawRecord, reconstruct_joint
from utils import bio_to_spans, spans_to_bio


def test_bio_span_roundtrip():
    spans = [(1, 3, "POS"), (5, 6, "NEG")]
    tags = spans_to_bio(8, spans)
    assert tags[1] == "B-POS" and tags[2] == "I-POS"
    assert tags[5] == "B-NEG"
    assert bio_to_spans(tags) == spans


def test_bio_decode_defensive():
    # I- without preceding B- starts a span; polarity switch splits.
    tags = ["I-POS", "I-POS", "B-NEG", "O"]
    spans = bio_to_spans(tags)
    assert spans == [(0, 2, "POS"), (2, 3, "NEG")]


def test_reconstruct_joint_from_masked_records():
    # Two aspects in one tweet: "Klay Thompson beats Warriors"
    recs = [
        RawRecord("$T$ beats Warriors", "Klay Thompson", "POS", "img1.jpg"),
        RawRecord("Klay Thompson beats $T$", "Warriors", "NEG", "img1.jpg"),
    ]
    insts = reconstruct_joint(recs)
    assert len(insts) == 1
    inst = insts[0]
    assert inst.tokens == ["Klay", "Thompson", "beats", "Warriors"]
    assert (0, 2, "POS") in inst.aspects
    assert (3, 4, "NEG") in inst.aspects
    # synthesized BIO
    assert inst.bio == ["B-POS", "I-POS", "O", "B-NEG"]

from src.data_pipeline import score_usda_match


def _candidate(description: str, data_type: str) -> dict:
    return {"description": description, "dataType": data_type}


def test_foundation_beats_branded_for_brown_sugar():
    foundation = _candidate("Brown sugar", "Foundation")
    branded = _candidate("C&H Pure Cane Brown Sugar", "Branded")
    assert score_usda_match("brown sugar", foundation) > score_usda_match("brown sugar", branded)


def test_sr_legacy_beats_branded():
    sr = _candidate("Olive oil, salad or cooking", "SR Legacy")
    branded = _candidate("Olive Oil Extra Virgin Premium Brand", "Branded")
    assert score_usda_match("olive oil", sr) > score_usda_match("olive oil", branded)


def test_exact_token_match_boosts_score():
    high = _candidate("chicken breast raw", "Branded")
    low = _candidate("turkey breast raw", "Branded")
    assert score_usda_match("chicken breast", high) > score_usda_match("chicken breast", low)


def test_score_range():
    c = _candidate("brown sugar raw cane", "Foundation")
    s = score_usda_match("brown sugar", c)
    assert 0.0 <= s <= 1.0


def test_unknown_datatype_treated_as_branded():
    c = _candidate("oats rolled", "Unknown")
    s = score_usda_match("oats", c)
    assert 0.0 <= s <= 1.0

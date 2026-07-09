from mini_vectordb.filters import matches_filter

def test_empty_filter_matches_everything():
    assert matches_filter({"category":"fruit"},{}) is True

def test_single_key_match():
    assert matches_filter({"category":"fruit"}, {"category":"fruit"}) is True

def test_single_key_mismatch():
    assert matches_filter({"category": "fruit"}, {"category": "animal"}) is False


def test_multiple_keys_all_match():
    metadata = {"category": "fruit", "source": "wiki"}
    filter_dict = {"category": "fruit", "source": "wiki"}
    assert matches_filter(metadata, filter_dict) is True


def test_multiple_keys_one_mismatch_fails_and_logic():
    metadata = {"category": "fruit", "source": "wiki"}
    filter_dict = {"category": "fruit", "source": "reddit"}
    assert matches_filter(metadata, filter_dict) is False


def test_missing_key_excluded():
    metadata = {"category": "fruit"}
    filter_dict = {"source": "wiki"}
    assert matches_filter(metadata, filter_dict) is False
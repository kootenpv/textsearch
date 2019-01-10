from textsearch import TextSearch, TSResult


def test_ignore_match():
    ts = TextSearch("ignore", "match")
    ts.add("hi")
    assert ts.extract("hi") == ["hi"]
    assert ts.extract("HI") == ["hi"]
    assert ts.extract("asdf") == []


def test_ignore_norm():
    ts = TextSearch("ignore", "norm")
    ts.add("hi", "HI")
    assert ts.extract("hi") == ["HI"]
    assert ts.extract("asdf") == []


def test_insensitive_match():
    ts = TextSearch("insensitive", "match")
    ts.add("hi")
    assert ts.extract("HI") == ["HI"]


def test_insensitive_object():
    ts = TextSearch("insensitive", "TSResult")
    ts.add("hi")
    assert ts.extract("HI")[0].end == 2


def test_sensitive_match():
    ts = TextSearch("sensitive", "TSResult")
    ts.add("hi")
    assert ts.extract("hi")
    assert not ts.extract("HI")


def test_smart_match():
    ts = TextSearch("smart", "TSResult")
    ts.add("hi")
    assert ts.extract("hi")[0].case == "lower"
    assert ts.extract("hi")[0].is_exact
    assert ts.extract("HI")[0].case == "upper"
    assert not ts.extract("HI")[0].is_exact
    assert ts.extract("Hi")[0].case == "title"
    assert not ts.extract("Hi")[0].is_exact
    ts.add("hI")
    assert ts.extract("hI")[0].case == "mixed"
    assert ts.extract("hI")[0].is_exact


def test_add_list():
    ts = TextSearch("smart", "match")
    ts.add(["hi", "bye", "hello"])
    assert ts.extract("hi bye hello") == ["hi", "bye", "hello"]


def test_add_list():
    ts = TextSearch("smart", "norm")
    ts.add({"hi": "greeting", "bye": "bye", "goodbye": "bye"})
    assert ts.extract("hi bye goodbye") == ["greeting", "bye", "bye"]

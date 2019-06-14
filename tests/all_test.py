import json
from textsearch import TextSearch


def test_ignore_match():
    ts = TextSearch("ignore", "match")
    ts.add("hi")
    assert ts.findall("hi") == ["hi"]
    assert ts.findall("HI") == ["hi"]
    assert ts.findall("asdf") == []


def test_ignore_norm():
    ts = TextSearch("ignore", "norm")
    ts.add("hi", "HI")
    assert ts.findall("hi") == ["HI"]
    assert ts.findall("asdf") == []


def test_insensitive_match():
    ts = TextSearch("insensitive", "match")
    ts.add("hi")
    assert ts.findall("HI") == ["HI"]


def test_insensitive_object():
    ts = TextSearch("insensitive", "object")
    ts.add("hi")
    assert ts.findall("HI")[0].end == 2


def test_sensitive_match():
    ts = TextSearch("sensitive", "object")
    ts.add("hi")
    assert ts.findall("hi")
    assert not ts.findall("HI")


def test_smart_match():
    ts = TextSearch("smart", "object")
    ts.add("hi")
    assert ts.findall("hi")[0].case == "lower"
    assert ts.findall("hi")[0].is_exact
    assert ts.findall("HI")[0].case == "upper"
    assert not ts.findall("HI")[0].is_exact
    assert ts.findall("Hi")[0].case == "title"
    assert not ts.findall("Hi")[0].is_exact
    ts.add("hI")
    assert ts.findall("hI")[0].case == "mixed"
    assert ts.findall("hI")[0].is_exact


def test_add_list():
    ts = TextSearch("smart", "match")
    ts.add(["hi", "bye", "hello"])
    assert ts.findall("hi bye hello") == ["hi", "bye", "hello"]


def test_add_dict():
    ts = TextSearch("smart", "norm")
    ts.add({"hi": "greeting", "bye": "bye", "goodbye": "bye"})
    assert ts.findall("hi bye goodbye") == ["greeting", "bye", "bye"]


def test_replace():
    ts = TextSearch("sensitive", "norm")
    ts.add("hi", "HI")
    assert ts.replace("test hi test") == "test HI test"


def test_replace_insensitive_keep_casing():
    ts = TextSearch("insensitive", "norm")
    ts.add("hi", "bye")
    assert ts.replace("test Hi test") == "test Bye test"
    assert ts.replace("test HI test") == "test BYE test"


def test_serializable():
    ts = TextSearch("sensitive", dict)
    ts.add("hi")
    result = ts.findall("hi")
    assert result
    assert json.dumps(result)


def test_http():
    ts = TextSearch("ignore", "norm")
    ts.add_http_handler(keep_result=True)
    assert ts.findall("http://google.com") == ["http://google.com"]


def test_http_no_keep():
    ts = TextSearch("ignore", "norm")
    ts.add_http_handler(keep_result=False)
    ts.add("google")
    assert ts.findall("http://google.com") == []


def test_twitter():
    ts = TextSearch("ignore", "norm")
    ts.add_twitter_handler(keep_result=True)
    assert ts.findall("@hello") == ["@hello"]
    assert ts.findall("#hello") == ["#hello"]


def test_custom_handler():
    def custom_handler(text, start, stop, norm):
        return start, stop, text[start:stop] + " is OK"

    ts = TextSearch("ignore", "norm", handlers=[("HI", True, custom_handler)])
    ts.add("hi", "HI")
    assert ts.findall("hi HI") == ['hi is OK', 'HI is OK']


def test_ignore_contains_word():
    ts = TextSearch("ignore", "norm")
    ts.add("hi", "HI")
    assert "hi" in ts
    assert "HI" in ts


def test_sensitive_contains_word():
    ts = TextSearch("sensitive", "norm")
    ts.add("hi", "HI")
    assert "hi" in ts
    assert "HI" not in ts


def test_contains():
    ts = TextSearch("sensitive", "norm")
    ts.add("hi", "HI")
    assert ts.contains("hi")


def test_to_ts():
    ts = TextSearch("sensitive", "norm")
    ts.add("hi")
    assert "hi" in ts.to_ts(TextSearch)


def test_sensitive_remove():
    ts = TextSearch("sensitive", "norm")
    ts.add("hi")
    assert len(ts) == 1
    ts.remove("hi")
    assert not len(ts)


def test_ignore_remove():
    ts = TextSearch("ignore", "norm")
    ts.add("hi")
    assert len(ts) == 1
    ts.remove("hi")
    assert not len(ts)


def test_smart_remove():
    ts = TextSearch("smart", "norm")
    ts.add("hi")
    assert len(ts) == 3
    ts.remove("hi")
    assert not len(ts)


def test_fast_no_bounds():
    ts = TextSearch("sensitive", "match", set(), set())
    ts.add("hi")
    assert ts.findall("asdfhiadsfs")


def test_left_bounds():
    ts = TextSearch("sensitive", "match")
    ts.add("hi")
    assert not ts.findall("asfdhi")


def test_right_bounds():
    ts = TextSearch("sensitive", "match")
    ts.add("hi")
    assert not ts.findall("hiasf")


def test_merge():
    ts1 = TextSearch("sensitive", "match")
    ts2 = TextSearch("sensitive", "match")
    ts1.add("hi")
    ts2.add("hi")
    assert len(ts1 + ts2) == 1
    ts1.remove("hi")
    ts2.add("bye")
    assert len(ts1 + ts2) == 2


def test_merge_handler():
    ts1 = TextSearch("sensitive", "norm")
    ts2 = TextSearch("sensitive", "norm")
    ts1.add_http_handler(True)
    assert (ts1 + ts2).handlers


def test_repr():
    assert repr(TextSearch("ignore", "match"))
    assert repr(TextSearch("ignore", "match", set(), set()))


def test_not_overlap():
    ts = TextSearch("ignore", "norm")
    ts.add("http://")
    ts.add_http_handler(True)
    assert len(ts.findall("https://vks.ai")) == 1


def test_not_overlap_2():
    ts = TextSearch("ignore", "norm")
    ts.add("hi", "HI")
    ts.add("hi hi", "h h")
    assert ts.replace("hi hi") == "h h"


def test_not_overlap_3():
    ts = TextSearch("ignore", "norm")
    ts.add("a")
    ts.add("a a")
    assert ts.findall("a a a") == ["a a", "a"]


def test_overlap():
    ts = TextSearch("ignore", "norm")
    ts.add("hi")
    ts.add("hi hi")
    assert len(ts.find_overlapping("hi hi")) == 3


def test_postfix_regex():
    ts = TextSearch("ignore", "norm")
    ts.add_regex_handler(["products"], r"\d+ ", keep_result=True, prefix=False)
    assert ts.findall("90 products") == ["90 products"]


def test_foreign_chars():
    ts = TextSearch("ignore", "norm", replace_foreign_chars=True)
    ts.add("á", "A")
    assert "a" in ts
    assert "á" in ts
    assert ts.contains("a")
    assert ts.contains("á")
    assert ts.findall("a")
    assert ts.findall("á")
    assert ts.find_overlapping("a")
    assert ts.find_overlapping("á")
    assert ts.replace("a") == "A"
    assert ts.replace("á") == "A"


def test_regex_norm():
    ts = TextSearch("insensitive", "norm")
    ts.add_regex_handler(["last "], r"\d", keep_result=True)
    assert ts.findall("last 5") == ["last 5"]


def test_regex_object():
    ts = TextSearch("insensitive", "object")
    ts.add_regex_handler(["last "], r"\d", keep_result=True)
    assert ts.findall("last 5")[0].norm == "last 5"


def test_regex_overlap():
    ts = TextSearch("insensitive", "object")
    ts.add_regex_handler(["last "], r"\d", keep_result=True)
    ts.add("last")
    assert ts.findall("last 5")[0].norm == "last 5"

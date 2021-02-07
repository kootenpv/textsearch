<p align="center">
  <img src="./resources/tslogo.png"/>
</p>

# textsearch

[![Build Status](https://travis-ci.org/kootenpv/textsearch.svg?branch=master)](https://travis-ci.org/kootenpv/textsearch)
[![Coverage Status](https://coveralls.io/repos/github/kootenpv/textsearch/badge.svg?branch=master)](https://coveralls.io/github/kootenpv/textsearch?branch=master)
[![PyPI](https://img.shields.io/pypi/v/textsearch.svg?style=flat-square)](https://pypi.python.org/pypi/textsearch/)
[![PyPI](https://img.shields.io/pypi/pyversions/textsearch.svg?style=flat-square)](https://pypi.python.org/pypi/textsearch/)

Find and/or replace multiple strings in text; focusing on convenience and using C for speed.

It mainly helps with providing convenience for NLP / text search related tasks.
For example, it will help find tokens by default only if it is a full word match (and not a sub-match).

### Features

- ✓ Compared to equivalent in regex, usually ~30-100x faster
- ✓ Tokenizer, string replacer, spell checker and many more things can be built on this (stay-tuned)
- ✓ Extendable (write your own handlers)
- ✓ Supports a prefix- or postfix based regex, usually something missing with other matchers
- ✓ Few depencies (relies on a C-module, credits to [WojciechMula/pyahocorasick](WojciechMula/pyahocorasick))
- ✓ Similar to [flashtext](https://github.com/vi3k6i5/flashtext) (by my friend Vikash), but 30% faster, as convenient, and more features
- ✓ Optional support for accented characters (~15% slowdown)
- ✓ Lots of [tests](test/all_test.py) (good place to see examples for inspiration) and good coverage

### Showcase

Here a list of the projects that are depending on `textsearch`.

|Name|Explanation|By|
|---------------------------------|--------------------------|--------------------|
| [rebrand](https://github.com/kootenpv/rebrand) |  Refactor your software using programming language independent string replacement. |  [kootenpv](https://github.com/kootenpv) |
| [contractions](https://github.com/kootenpv/contractions) |  Fixes contractions such as `you're` to you `are` |  [kootenpv](https://github.com/kootenpv) |
| [tok](https://github.com/kootenpv/tok) |  Find strings/words in text; convenience and C speed |  [kootenpv](https://github.com/kootenpv) |


### Installation

Works on Python 3+ (make an issue if you *really* need Python 2)

    pip install textsearch

### Main functionality

```python
ts = TextSearch("ignore", "norm")
ts.add("hi", "HI")
"hi" in ts            # True
ts.contains("hi!")    # True
ts.replace("hi!")     # "HI!"
ts.remove("hi")
ts.contains("hi!")    # False
ts
TextSearch(case='ignore', returns='norm', num_items=0)
```

### More examples

See [tests/](tests/all_test.py) for more usage examples.

```python
from textsearch import TextSearch

ts = TextSearch(case="ignore", returns="match")
ts.add("hi")
ts.findall("hello, hi")
# ["hi"]

ts = TextSearch(case="ignore", returns="norm")
ts.add("hi", "greeting")
ts.add("hello", "greeting")
ts.findall("hello, hi")
# ["greeting", "greeting"]

ts = TextSearch(case="ignore", returns="match")
ts.add(["hi", "bye"])
ts.findall("hi! bye! HI")
# ["hi", "bye", "hi"]

ts = TextSearch(case="insensitive", returns="match")
ts.add(["hi", "bye"])
ts.findall("hi! bye! HI")
# ["hi", "bye", "HI"]

ts = TextSearch("sensitive", "object")
ts.add("HI")
ts.findall("hi")
# []
ts.findall("HI")
# [TSResult(match='HI', norm='HI', start=0, end=2, case='upper', is_exact=True)]

ts = TextSearch("sensitive", dict)
ts.add("hI")
ts.findall("hI")
# [{'case': 'mixed', 'end': 2, 'exact': True, 'match': 'hI', 'norm': 'hI', 'start': 0}]
```

### Usage

`TextSearch` takes the following arguments:

    case: one of "ignore", "insensitive", "sensitive", "smart"
        - ignore: converts both sought words and to-be-searched to lower case before matching.
                  Text matches will always be returned in lowercase.
        - insensitive: converts both sought words and to-be-searched to lower case before matching.
                       However: it will return the original casing as it uses the position found.
        - sensitive: does not do any conversion, will only match on exact words added.
        - smart: takes an input `k` and also adds k.title(), k.upper() and k.lower(). Matches sensitively.
    returns: one of 'match', 'norm', 'object' (becomes TSResult) or a custom class
        See the examples!
        - match: returns the sought key when a match occurs
        - norm: returns the associated (usually normalized) value when a key match occurs
        - object: convenient object for working with matches, e.g.:
          TSResult(match='HI', norm='greeting', start=0, end=2, case='upper', is_exact=True)
        - class: bring your own class that will get instantiated like:
          MyClass(**{"match": k, "norm": v, "case": "lower", "exact": False, start: 0, end: 1})
          Trick: to get json-serializable results, use `dict`.
    left_bound_chars (set(str)):
        Characters that will determine the left-side boundary check in findall/replace.
        Defaults to set([A-Za-z0-9_])
    right_bound_chars (set(str)):
        Characters that will determine the right-side boundary check in findall/replace
        Defaults to set([A-Za-z0-9_])
    replace_foreign_chars (default=False): replaces 'á' with 'a' for example in both input and target.
        Adds a roughly 15% slowdown.
    handlers (list): provides a way to add hooks to matches.
        Currently only used when left and/or right bound chars are set.
        Regex can only be used when using norm
        The default handler that gets added in any case will check boundaries.
        Check how to conveniently add regex at the `add_regex_handler` function.
        Default: [(False, True, self.bounds_check)]
        - The first argument should be the normalized tag to fire on.
        - The second argument should be whether to keep the result
        - The third should be the handler function, that takes the arguments:
          - text: the original sentence/document text
          - start: the starting position of said string
          - stop: the ending position of said string
          - norm: the normalized result found
          Should return:
          - start: In case start position should change it is possible
          - stop: In case end position should change it is possible
          - norm: the new returned item. In case this is None, it will be removed
        Custom example:
          >>> def custom_handler(text, start, stop, norm):
          >>>    return start, stop, text[start:stop] + " is OK"
          >>> ts = TextSearch("ignore", "norm", handlers=[("HI", True, custom_handler)])
          >>> ts.add("hi", "HI")
          >>> ts.findall("hi HI")
          ['hi is OK', 'HI is OK']

### Benchmarks

Coming soon, will compare what's available in Java and Go as well, and flashtext, regex.

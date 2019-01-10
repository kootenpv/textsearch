# textsearch

Find strings/words in text; convenience and C speed

```python
from textsearch import TextSearch
ts = TextSearch(case="ignore", returns="match")
ts.add("hi")
ts.extract("hi hi hi")
# ["hi", "hi", "hi"]
```

See [tests/](tests/) or the doc string of `TextSearch` for more examples.

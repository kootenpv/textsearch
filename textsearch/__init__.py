import re
import string
from anyascii import anyascii
import ahocorasick

ALPHANUM = set(string.digits + string.ascii_letters + '_')
ALPHA = set(string.ascii_letters)
ALPHA_LOWER = set(string.ascii_lowercase)
ALPHA_UPPER = set(string.ascii_uppercase)


class TSResult(object):
    __slots__ = ('match', 'norm', 'start', 'end', 'case', 'is_exact')

    def __init__(self, match, norm, start, end, case=None, exact=None):
        self.match = match
        self.norm = norm
        self.start = start
        self.end = end
        self.case = case
        self.is_exact = exact

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(["{}={}".format(x, repr(getattr(self, x))) for x in self.__slots__]),
        )


def to_sentence_case(k):
    return k[0].upper() + k[1:].lower()


def determine_case(word):
    if word == word.upper():
        return "upper"
    if word == word.title():
        return "title"
    if word == word.lower():
        return "lower"
    if word == to_sentence_case(word):
        return "sent"
    return "mixed"


case_fn = {
    "upper": str.upper,
    "lower": str.lower,
    "title": str.title,
    "sent": to_sentence_case
    # else: cover with .get(, lambda x: x)
}


class TextSearch(object):
    def __init__(
        self,
        case,
        returns,
        left_bound_chars=ALPHANUM,
        right_bound_chars=ALPHANUM,
        replace_foreign_chars=False,
        handlers=None,
        **kwargs
    ):
        """ TextSearch is built on a C implementation of Aho-Corasick available in the ahocorasick[0] lib.

        It mainly helps with providing convenience for NLP / text search related tasks.
        For example, it will help find tokens by default only if it is a full word match (and not a sub-match).
        Though this can be adapted.

        Arguments:
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
                - "object": convenient object for working with matches, e.g.:
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
                Default: (False, True, self.bounds_check)
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
        Examples:
            >>> from textsearch import TextSearch

            >>> ts = TextSearch(case="ignore", returns="match")
            >>> ts.add("hi")
            >>> ts.findall("hello, hi")
            ["hi"]

            >>> ts = TextSearch(case="ignore", returns="norm")
            >>> ts.add("hi", "greeting")
            >>> ts.add("hello", "greeting")
            >>> ts.findall("hello, hi")
            ["greeting", "greeting"]

            >>> ts = TextSearch(case="ignore", returns="match")
            >>> ts.add(["hi", "bye"])
            >>> ts.findall("hi! bye! HI")
            ["hi", "bye", "hi"]

            >>> ts = TextSearch(case="insensitive", returns="match")
            >>> ts.add(["hi", "bye"])
            >>> ts.findall("hi! bye! HI")
            ["hi", "bye", "HI"]

            >>> ts = TextSearch("sensitive", "object")
            >>> ts.add("HI")
            >>> ts.findall("hi")
            []
            >>> ts.findall("HI")
            [TSResult(match='HI', norm='HI', start=0, end=2, case='upper', is_exact=True)]

            >>> ts = TextSearch("sensitive", dict)
            >>> ts.add("hI")
            >>> ts.findall("hI")
            [{'case': 'mixed', 'end': 2, 'exact': True, 'match': 'hI', 'norm': 'hI', 'start': 0}]

        Notes:
           [0]: https://github.com/WojciechMula/pyahocorasick
        """
        case_opts = ("ignore", "insensitive", "sensitive", "smart")
        if case not in case_opts:
            raise ValueError("argument 'case' must be one of {!r}".format(case_opts))
        returns_opts = ("match", "norm", "object")
        if isinstance(returns, str) and returns not in returns_opts:
            raise ValueError("argument 'returns' must be one of {!r} or class".format(returns_opts))
        if returns in (True, False, bool, int, list, tuple, set):  # dict is possible
            raise ValueError("argument 'returns' must be one of {!r} or class".format(returns_opts))

        self.automaton = ahocorasick.Automaton()
        # if http:
        #     self.add_http()
        self.case = case
        self.returns = returns if returns != "object" else TSResult
        self.extract_fn = self.get_extraction_fn()
        self._ignore_case_in_search = self.case in ["ignore", "insensitive"]
        self.left_bound_chars = left_bound_chars or set()
        self.right_bound_chars = right_bound_chars or set()
        self.words_changed = False
        self.replace_foreign_chars = replace_foreign_chars
        self.handlers = handlers or []
        self._root_dict = {}

    def __add__(self, other):
        if not isinstance(other, TextSearch):
            raise TypeError("Can only be merged with another TextSearch derived class")
        ts = self.to_ts(other)
        ts.add(self._root_dict)
        ts.add(other._root_dict)
        handlers = []
        seen = set()
        for x in other.handlers + self.handlers:
            if x[-1] not in seen:
                handlers.append(x)
                seen.add(x[-1])
        ts.handlers = handlers
        return ts

    def add_http_handler(self, keep_result):
        self.add_regex_handler(["http://", "https://", "www."], "[^ ]+", keep_result)

    def add_twitter_handler(self, keep_result):
        self.add_tweet_hashtag(keep_result)
        self.add_tweet_mention(keep_result)

    def add_tweet_hashtag(self, keep_result):
        self.add_regex_handler(["#"], "[a-zA-Z][^ !$%^&*+.]+", keep_result)

    def add_tweet_mention(self, keep_result):
        self.add_regex_handler(["@"], "[a-zA-Z][a-zA-Z0-9_]+", keep_result)

    def add_regex_handler(
        self, words, regex, keep_result, prefix=True, name=None, return_value=None, flags=0
    ):
        """ Allows adding a regex that should hit when a prefix or postfix gets found.
        words: list of words that when hit trigger the regex handler.
        regex: the regex pattern that gets applied to a matched word.
            NOTE: Already assumes the words to occur at either the start or the end of the match!
        keep_result: whether to throw away the result when found.
        prefix:
            - True means from the start of the match to the end of the whole string
            - False means means from the start of the whole string to the end of the match
        name: give a name for the token (used internally, only set to avoid a clash).
            By default would choose the first of the words, stripped of special chars.
        return_value: either a fixed value when set, otherwise the regex match.

        Let's explain this example
        ts.add_regex_handler(["http://", "https://", "www."], "[^ ]+", keep_result=True)

        Will match on http://, https://, www. and try the regex '[^ ]+' afterwards, matching urls.

        Warning: better not to try to use regex with too short of a prefix, e.g. just a single letter.
        """
        # if self.returns is not "norm":
        #     raise ValueError("Regex currently only works with returns='norm'")
        name = name or ("$" + (re.sub("[^a-zA-Z]", "", words[0]).upper() or words[0]))
        for word in words:
            self.add_one(word, name)
        handler = self.prefix_regex_handler if prefix else self.postfix_regex_handler
        self.handlers.append((name, keep_result, handler(regex, return_value, flags)))

    def to_ts(self, ts):
        block_keys = (
            "automaton",
            "_ignore_case_in_search",
            "words_changed",
            "extract_fn",
            "_root_dict",
        )
        data = self._root_dict
        if ts.__class__.__name__ == 'type':
            inits = {k: v for k, v in self.__dict__.items() if k not in block_keys}
            ts = ts(**inits)
        else:
            inits = {k: v for k, v in ts.__dict__.items() if k not in block_keys}
            ts = ts.__class__(**inits)
        ts.add(data)
        return ts

    def add(self, k, v=None):
        if isinstance(k, (set, list)):
            for x in k:
                self.add_one(x, v)
        elif isinstance(k, dict):
            for kk, vv in k.items():
                self.add_one(kk, vv)
        else:
            self.add_one(k, v)

    def add_ignore(self, k, v, length):
        if self.returns == "match":
            self.automaton.add_word(k.lower(), (length, k))
        elif self.returns == "norm":
            self.automaton.add_word(k.lower(), (length, v))
        else:
            raise ValueError(
                "ignore only returns a match or normalized value. Maybe you want insensitive?"
            )

    def add_insensitive(self, k, v, length):
        if self.returns == "norm":
            # raise ValueError("For returning simply a normalized value, use case='ignore'")
            # # self.automaton.add_word(k.lower(), v)
            self.automaton.add_word(k.lower(), (length, v))
        elif self.returns == "match":
            self.automaton.add_word(k.lower(), (length, -1))
        # adding object
        else:
            self.automaton.add_word(k.lower(), (length, {"norm": v, "exact": False}))

    def remove(self, k):
        """ Remove k from known words. Takes into account the casing. """
        if k not in self._root_dict:
            return False
        del self._root_dict[k]
        self.words_changed = True
        if self.replace_foreign_chars:
            k = anyascii(k)
        if self.case == "smart":
            self.automaton.remove_word(k)
            self.automaton.remove_word(k.lower())
            self.automaton.remove_word(k.title())
            self.automaton.remove_word(k.upper())
            self.automaton.remove_word(to_sentence_case(k))
        elif self.case == "sensitive":
            self.automaton.remove_word(k)
        else:
            self.automaton.remove_word(k.lower())
        return True

    def add_sensitive_string(self, k, v, length):
        case = determine_case(k)
        text_value = k if self.returns.startswith("match") else v
        result = text_value + "_" + case if self.returns.endswith("_case") else text_value

        self.automaton.add_word(k, (length, result))

        if self.case != "smart":
            return

        for key, case in zip(
            [k.upper(), k.title(), k.lower(), to_sentence_case(k)],
            ["upper", "title", "lower", "sent"],
        ):
            if key == k or key in self.automaton:
                continue
            self.automaton.add_word(key, (length, result))

    def add_sensitive_object(self, k, v, length):
        case = determine_case(k)

        obj = {"match": k, "norm": v, "case": case, "exact": True}
        self.automaton.add_word(k, (length, obj))

        if self.case != "smart":
            return

        for key, case in zip(
            [k.upper(), k.title(), k.lower(), to_sentence_case(k)],
            ["upper", "title", "lower", "sent"],
        ):
            if key == k or key in self.automaton:
                continue
            obj = {"match": key, "norm": v, "case": case, "exact": False}
            self.automaton.add_word(key, (length, obj))

    def add_sensitive(self, k, v, length):

        if self.returns in ["match", "norm"]:
            self.add_sensitive_string(k, v, length)

        else:  # going to pass some parameters to object
            self.add_sensitive_object(k, v, length)

    def add_one(self, k, v=None):
        self._root_dict[k] = v
        self.words_changed = True
        if self.replace_foreign_chars:
            k = anyascii(k)

        v = k if v is None else v
        length = len(k)

        if self.case == "ignore":
            self.add_ignore(k, v, length)
        elif self.case == "insensitive":
            self.add_insensitive(k, v, length)
        else:
            self.add_sensitive(k, v, length)

    def bounds_check(self, text, start, stop, norm):
        if len(text) != stop and text[stop] in self.right_bound_chars:
            return None, None, None
        # watch out... cannot just get index as we might risk -1
        if start != 0 and text[start - 1] in self.left_bound_chars:
            return None, None, None
        return start, stop, self.extract_fn(start, stop, norm, text)

    def contains(self, text):
        """ Test whether any known words match in text.
        text: str
        returns: bool
        """
        self.automaton.make_automaton()
        self.words_changed = False
        if self.replace_foreign_chars:
            text = anyascii(text)
        _text = text.lower() if self._ignore_case_in_search else text
        if not self.handlers and not self.left_bound_chars and not self.right_bound_chars:
            for end_index, (length, norm) in self.automaton.iter(_text):
                return True
        handlers = self.handlers + [(False, True, self.bounds_check)]
        for end_index, (length, norm) in self.automaton.iter(_text):
            if norm is None:
                continue
            start = end_index - length + 1
            stop = end_index + 1
            for compare, keep_result, handler in handlers:
                if compare and compare is not norm:
                    continue
                start, stop, result = handler(text, start, stop, norm)
                if result is not None and keep_result:
                    return True
        return False

    def build_automaton(self):
        if self.words_changed:
            self.automaton.make_automaton()
        self.words_changed = False

    def findall(self, text):
        """ Finds the known words in text.
        text: str
        returns: list of matches
        """
        self.build_automaton()
        if self.replace_foreign_chars:
            text = anyascii(text)
        keywords = []
        current_stop = -1
        _text = text.lower() if self._ignore_case_in_search else text
        # if not self.handlers and not self.left_bound_chars and not self.right_bound_chars:
        #     # might overlap?
        #     for end_index, (length, norm) in self.automaton.iter(_text):
        #         start = end_index - length + 1
        #         stop = end_index + 1
        #         res = self.extract_fn(end_index - length + 1, end_index + 1, norm, text)
        #         if start >= current_stop:
        #             current_stop = stop
        #             result = (current_stop - start, start, current_stop, res)
        #             keywords.append(result)
        #         elif stop - start > keywords[-1][0]:
        #             current_stop = max(stop, current_stop)
        #             result = (current_stop - start, start, current_stop, res)
        #             keywords[-1] = result
        #     return [x[3] for x in keywords]
        handlers = self.handlers + [(False, True, self.bounds_check)]
        for end_index, (length, norm) in self.automaton.iter(_text):
            if norm is None:
                continue
            start = end_index - length + 1
            stop = end_index + 1
            for compare, keep_result, handler in handlers:
                if compare and compare is not norm and norm != {"norm": compare, "exact": False}:
                    continue
                start, stop, result = handler(text, start, stop, norm)
                if result is None:  # maybe want to remove this
                    break
                if not keep_result:
                    current_stop = stop
                    keywords.append((current_stop - start, start, current_stop, None))
                    break
                if start >= current_stop:
                    current_stop = stop
                    result = (current_stop - start, start, current_stop, result)
                    keywords.append(result)
                elif stop - start > keywords[-1][0]:
                    current_stop = max(stop, current_stop)
                    result = (current_stop - start, start, current_stop, result)
                    keep_up_to_ind = next((i for i, k in enumerate(keywords) if k[1] == start),
                                          -1)  # find the first index of a keyword with the same starting position
                    keywords = keywords[:keep_up_to_ind]  # keep keyword up to this index
                    keywords.append(result)  # append the current keyword that contains all the removed keywords
                    # whyyyyy ?? better commment it out
                    # else:
                    #     import pdb

                    #     pdb.set_trace()
                    #     current_stop = stop
                    #     result = (current_stop - start, start, current_stop, result)
                    #     keywords.append(result)
        return [x[3] for x in keywords if x[3] is not None]

    def find_overlapping(self, text):
        """ Finds the known words in text.
        text: str
        returns: list of matches
        """
        self.build_automaton()
        if self.replace_foreign_chars:
            text = anyascii(text)
        keywords = []
        _text = text.lower() if self._ignore_case_in_search else text
        if not self.handlers and not self.left_bound_chars and not self.right_bound_chars:
            return [
                self.extract_fn(end_index - length + 1, end_index + 1, norm, text)
                for end_index, (length, norm) in self.automaton.iter(_text)
            ]
        handlers = self.handlers + [(False, True, self.bounds_check)]
        for end_index, (length, norm) in self.automaton.iter(_text):
            if norm is None:
                continue
            start = end_index - length + 1
            stop = end_index + 1
            for compare, keep_result, handler in handlers:
                if compare and compare is not norm:
                    continue
                _, _, result = handler(text, start, stop, norm)
                if not keep_result:
                    continue
                if result is None:  # maybe want to remove this
                    continue
                keywords.append(result)
        return [x for x in keywords if x is not None]

    def prefix_regex_handler(self, r, return_value, flags=0):
        regex = re.compile(r, flags=flags)

        def regex_handler(text, start, stop, norm):
            # bounds check
            if start != 0 and text[start - 1] in self.left_bound_chars:
                return start, stop, None
            reg_res = regex.match(text[stop:])
            if reg_res:
                stop = stop + reg_res.end()
                rv = return_value or text[start:stop]
                if not isinstance(self.returns, str):
                    rv = {"norm": rv}
                rv = self.extract_fn(start, stop, rv, text)
                return start, stop, rv
            return start, stop, None

        return regex_handler

    def postfix_regex_handler(self, r, return_value, flags=0):
        regex = re.compile(r, flags=flags)

        def regex_handler(text, start, stop, norm):
            # bounds check
            if len(text) != stop and text[stop] in self.right_bound_chars:
                return start, stop, None
            reg_res = regex.search(text[:start])
            if reg_res:
                start = reg_res.start()
                rv = return_value or text[start:stop]
                if not isinstance(self.returns, str):
                    rv = {"norm": rv}
                rv = self.extract_fn(start, stop, rv, text)
                return start, stop, rv
            return start, stop, None

        return regex_handler

    def replace(self, text, return_entities=False):
        """ Replaces known words in text.
        text: str
        returns: replaced str
        If return_entities=True, returns: replaced str, list of matches
        """
        if self.returns != "norm" and not callable(self.returns):
            raise ValueError("no idea how i would do that")
        self.build_automaton()
        if self.replace_foreign_chars:
            text = anyascii(text)
        keywords = [(None, None, 0, ("", ""))]
        current_stop = -1
        _text = text.lower() if self._ignore_case_in_search else text
        handlers = self.handlers + [(False, True, self.bounds_check)]
        for end_index, (length, norm) in self.automaton.iter(_text):
            start = end_index - length + 1
            stop = end_index + 1
            for compare, keep_result, handler in handlers:
                if compare and compare is not norm:
                    continue
                start, stop, result = handler(text, start, stop, norm)
                if result is None:  # maybe want to remove this
                    break
                if not keep_result:
                    break
                if start >= current_stop:
                    current_stop = stop
                    result = (current_stop - start, start, current_stop, (text[start:stop], result))
                    keywords.append(result)
                elif stop - start > keywords[-1][0]:
                    current_stop = max(current_stop, stop)
                    result = (current_stop - start, start, current_stop, (text[start:stop], result))
                    # keywords[-1] = (stop - start, start, stop, result)
                    keywords[-1] = result
        keywords.append((None, len(text), None, ("", "")))
        text_ = ""
        for (_, start1, stop1, result1), (_, start2, stop2, result2) in zip(
            keywords[:-1], keywords[1:]
        ):
            norm = result2[1] if isinstance(result2[1], str) else result2[1].norm
            text_ += text[stop1:start2] + norm
        if return_entities:
            return text_, [x[-1] for x in keywords[1:-1]]
        return text_

    def extract_str(self, start_index, end, result, text):
        return result

    def extract_insensitive_match(self, start, end, result, text):
        match = text[start:end]
        return match

    def extract_insensitive_norm(self, start, end, result, text):
        case = determine_case(text[start:end])
        return case_fn.get(case, lambda x: x)(result)

    def extract_insensitive_object(self, start, end, result, text):
        match = text[start:end]
        result["match"] = match
        result["case"] = determine_case(match)
        result["start"] = start
        result["end"] = end
        return self.returns(**result)

    def extract_object(self, start, end, result, text):
        result["start"] = start
        result["end"] = end
        return self.returns(**result)

    def get_extraction_fn(self):
        if self.case == "insensitive":
            if self.returns == "match":
                extract_fn = self.extract_insensitive_match
            elif self.returns == "norm":
                extract_fn = self.extract_insensitive_norm
            else:
                extract_fn = self.extract_insensitive_object
        elif isinstance(self.returns, str):
            extract_fn = self.extract_str
        else:
            extract_fn = self.extract_object
        return extract_fn

    def __len__(self):
        return len(self.automaton)

    def __iter__(self):
        return self.automaton.__iter__()

    def __contains__(self, key):
        # note that smart includes lowercase, so it's an easy check
        if self.replace_foreign_chars:
            key = anyascii(key)
        if self._ignore_case_in_search or self.case == "smart":
            return key.lower() in self.automaton
        return key in self.automaton

    def __repr__(self):
        s = ["num_items={}".format(len(self))]
        if self.left_bound_chars != ALPHANUM:
            s.append("{}={!r}".format("left_bound_chars", self.left_bound_chars))
        if self.right_bound_chars != ALPHANUM:
            s.append("{}={!r}".format("right_bound_chars", self.left_bound_chars))
        if self.replace_foreign_chars:
            s.append("replace_foreign_chars=True")
        return "{}(case={!r}, returns={!r}, {})".format(
            self.__class__.__name__, self.case, self.returns, ", ".join(s)
        )

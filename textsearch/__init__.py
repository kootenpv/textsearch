import re
import string
import unidecode
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


def determine_case(word):
    if word == word.upper():
        return "upper"
    if word == word.title():
        return "title"
    if word == word.lower():
        return "lower"
    return "mixed"


HTTP_FIX = re.compile("https?://[^ ]+")


def prefix_regex_handler(r, flags=0):
    regex = re.compile(r, flags=flags)

    def regex_handler(text, start, stop, norm):
        reg_res = regex.match(text[start:])
        if reg_res:
            stop = start + reg_res.end()
            return start, stop, reg_res.group()
        return start, stop, None

    return regex_handler


def postfix_regex_handler(r, flags=0):
    regex = re.compile(r, flags=flags)

    def regex_handler(text, start, stop, norm):
        reg_res = regex.search(text[:stop])
        if reg_res:
            start = reg_res.start()
            return start, stop, reg_res.group()
        return start, stop, None

    return regex_handler


def handle_default(text, start, stop, norm):
    return start, stop, text[start:stop]


reg_res = re.compile("(https?://|ftp://|www.)[^ ]+")


def regex_handler(text, start, stop, norm):
    if reg_res:
        stop = start + reg_res.end()
        return stop, reg_res.group()
    return start, stop, None


class TextSearch(object):
    def __init__(
        self,
        case,
        returns,
        left_bound_chars=ALPHANUM,
        right_bound_chars=ALPHANUM,
        replace_foreign_chars=False,
        http=True,
        handlers=None,
    ):
        """ TextSearch is built on a C implementation of Aho-Corasick available in the ahocorasick[0] lib.

        It mainly helps with providing convenience for NLP / text search related tasks.
        For example, it will help find tokens by default only if it is a full word match (and not a sub-match).
        Though this can be adapted.

        Attributes:
            case: one of "ignore", "insensitive", "sensitive", "smart"
                - ignore: converts both sought words and to-be-searched to lower case before matching.
                          Text matches will always be returned in lowercase.
                - insensitive: converts both sought words and to-be-searched to lower case before matching.
                               However: it will return the original casing as it uses the position found.
                - sensitive: does not do any conversion, will only match on exact words added.
                - smart: takes an input `k` and also adds k.title(), k.upper() and k.lower(). Matches sensitively.
            returns: one of 'match', 'norm', 'match_case', 'norm_case', 'TSResult' (becomes TSResult) or class
                See the examples!
                - match: returns the sought key when a match occurs
                - norm: returns the associated (usually normalized) value when a key match occurs
                - match_case: returns the sought key when a match occurs with case added
                - norm_case: returns the associated (usually normalized) value when a key match occurs + case
                - TSResult (str): convenient object for working with matches, e.g.:
                  TSResult(match='HI', norm='greeting', start=0, end=2, case='upper', is_exact=True={})
                - class: bring your own class that will get instantiated like:
                  MyClass(**{"match": k, "norm": v, "case": "lower", "exact": False, start: 0, end: 1})
            left_bound_chars (set(str)): Characters that will determine the left-side boundary check in extract/replace.
                Defaults to set([A-Za-z0-9_])
            right_bound_chars (set(str)): Characters that will determine the right-side boundary check in extract/replace
                Defaults to set([A-Za-z0-9_])
            replace_foreign_chars (default=False): replaces 'á' with 'a' for example in both input and target.
                Adds a roughly 15% slowdown.

        Examples:
            >>> from textsearch import TextSearch

            >>> ts = TextSearch(case="ignore", returns="match")
            >>> ts.add("hi")
            >>> ts.extract("hello, hi")
            ["hi"]

            >>> ts = TextSearch(case="ignore", returns="norm")
            >>> ts.add("hi", "greeting")
            >>> ts.add("hello", "greeting")
            >>> ts.extract("hello, hi")
            ["greeting", "greeting"]

            >>> ts = TextSearch(case="ignore", returns="match")
            >>> ts.add(["hi", "bye"])
            >>> ts.extract("hi! bye! HI")
            ["hi", "bye", "hi"]

            >>> ts = TextSearch(case="insensitive", returns="match")
            >>> ts.add(["hi", "bye"])
            >>> ts.extract("hi! bye! HI")
            ["hi", "bye", "HI"]

            >>> ts = TextSearch(case="smart", returns="norm_case")
            >>> ts.add({"hi": "greeting", "bye": "greeting"})
            >>> ts.extract("Hi! bye!")
            ['greeting_title', 'greeting_lower']

            >>> ts = TextSearch("sensitive", "TSResult")
            >>> ts.add("HI")
            >>> ts.extract("hi")
            []
            >>> ts.extract("HI")
            [TSResult(match='HI', norm='HI', start=0, end=2, case='upper', is_exact=True={})]

            >>> ts = TextSearch("sensitive", dict)
            >>> ts.add("hI")
            >>> ts.extract("hI")
            [{'case': 'mixed', 'end': 2, 'exact': True, 'match': 'hI', 'norm': 'hI', 'start': 0}]

        Notes:
           [0]: https://github.com/WojciechMula/pyahocorasick
        """
        case_opts = ("ignore", "insensitive", "sensitive", "smart")
        if case not in case_opts:
            raise ValueError("argument 'case' must be one of {!r}".format(case_opts))
        returns_opts = ("match", "norm", "match_case", "norm_case", "TSResult")
        if isinstance(returns, str) and returns not in returns_opts:
            raise ValueError("argument 'returns' must be one of {!r} or class".format(returns_opts))
        if returns in (True, False, bool, int, list, tuple, set):  # dict is possible
            raise ValueError("argument 'returns' must be one of {!r} or class".format(returns_opts))

        self.automaton = ahocorasick.Automaton()
        self.http = http
        # if http:
        #     self.add_http()
        self.case = case
        self.returns = returns if returns != "TSResult" else TSResult
        self.extract_fn = self.get_extraction_fn()
        self._ignore_case_in_search = self.case in ["ignore", "insensitive"]
        self.left_bound_chars = left_bound_chars or set()
        self.right_bound_chars = right_bound_chars or set()
        self.words_changed = False
        self.replace_foreign_chars = replace_foreign_chars
        self.handlers = handlers or []
        self._root_dict = {}

    def add_http_handler(self, keep_result):
        self.add_regex_handler(
            ["http://", "https://", "www."], "({words})[^ ]+", prefix_regex_handler, keep_result
        )

    def add_regex_handler(self, words, regex, keep_result, prefix=True):
        name = "$" + re.sub("[^a-zA-Z]", "", words[0]).upper()
        for word in words:
            self.automaton.add_word(word, (len(word), name))
        regex = regex.format(words="|".join([re.escape(x) for x in words]))
        handler = prefix_regex_handler if prefix else postfix_regex_handler
        self.handlers.append((name, keep_result, handler(regex)))

    def merge(self, ts, complain=True):
        print("OK")

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
        ts.add(data)
        return ts

    def add(self, k, v=None):
        if isinstance(k, list):
            for x in k:
                self.add_one(x, v)
        elif isinstance(k, dict):
            for k, v in k.items():
                self.add_one(k, v)
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
            raise ValueError("For returning simply a normalized value, use case='ignore'")
            # self.automaton.add_word(k.lower(), v)
        elif self.returns == "match":
            self.automaton.add_word(k.lower(), (length, -1))
        elif self.returns == "match_case":
            self.automaton.add_word(k.lower(), (length, -1))
        elif self.returns == "norm_case":
            self.automaton.add_word(k.lower(), (length, {"norm": v}))
        # adding object
        else:
            self.automaton.add_word(k.lower(), (length, {"norm": v, "exact": False}))

    def add_sensitive_string(self, k, v, length):
        case = determine_case(k)
        text_value = k if self.returns.startswith("match") else v
        result = text_value + "_" + case if self.returns.endswith("_case") else text_value

        self.automaton.add_word(k, (length, result))

        if self.case != "smart":
            return

        for key, case in zip([k.upper(), k.title(), k.lower()], ["upper", "title", "lower"]):
            if key == k or key in self.automaton:
                continue
            self.automaton.add_word(key, (length, text_value + "_" + case))

    def add_sensitive_object(self, k, v, length):
        case = determine_case(k)

        obj = {"match": k, "norm": v, "case": case, "exact": True}
        self.automaton.add_word(k, (length, obj))

        if self.case != "smart":
            return

        for key, case in zip([k.upper(), k.title(), k.lower()], ["upper", "title", "lower"]):
            if key == k or key in self.automaton:
                continue
            obj = {"match": key, "norm": v, "case": case, "exact": False}
            self.automaton.add_word(key, (length, obj))

    def add_sensitive(self, k, v, length):

        if self.returns in ["match", "match_case", "norm", "norm_case"]:
            self.add_sensitive_string(k, v, length)

        else:  # going to pass some parameters to object
            self.add_sensitive_object(k, v, length)

    def add_one(self, k, v=None):
        self._root_dict[k] = v
        self.words_changed = True
        if self.replace_foreign_chars:
            k = unidecode.unidecode(k)

        v = k if v is None else v
        length = len(k)

        if self.case == "ignore":
            self.add_ignore(k, v, length)
        elif self.case == "insensitive":
            self.add_insensitive(k, v, length)
        else:
            self.add_sensitive(k, v, length)

    def add_tokenization(self):
        ignore = (None, None)
        self.add("$", ignore)
        # self.add(" !", " ! ")
        for s in "!.?-":
            for i in range(1, 10):
                if i == 1 and s == "-":
                    continue
                c = s * i
                e = s * 3 if i > 1 else s
                end = "\n" if i == 1 or s != "-" else " "
                self.add("{}".format(c), ignore)

        for i in range(1, 10):
            self.add("\n" * i, ignore)

        self.add("- ", ignore)

        self.add("...", ignore)

        for x in string.ascii_lowercase:
            self.add(" " + x + ".", ignore)

        for x in "0123456789":
            self.add(x + ",", ignore)
            self.add(x + ".", ignore)
            self.add("$" + x + ",", ignore)
            self.add("$" + x + ".", ignore)

        # self.add(".", ignore)
        self.add(",", ignore)

        # self.add("cannot", "can not")
        # self.add("can't", "can n't")

    def bounds_check(self, text, start, stop, norm):
        if len(text) != stop and text[stop] in self.right_bound_chars:
            return None, None, None
        # watch out... cannot just get index as we might risk -1
        if start != 0 and text[start - 1] in self.left_bound_chars:
            return None, None, None
        return start, stop, self.extract_fn(start, stop, norm, text)

    def contains(self, text):
        self.automaton.make_automaton()
        self.words_changed = False
        if self.replace_foreign_chars:
            text = unidecode.unidecode(text)
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

    def findall(self, text):
        self.automaton.make_automaton()
        self.words_changed = False
        if self.replace_foreign_chars:
            text = unidecode.unidecode(text)
        keywords = []
        current_stop = -1
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
                start, stop, result = handler(text, start, stop, norm)
                if result is None:  # maybe want to remove this
                    break
                if not keep_result:
                    break
                if start >= current_stop:
                    current_stop = stop
                    result = (current_stop - start, start, current_stop, result)
                    keywords.append(result)
                elif stop - start > keywords[-1][0]:
                    current_stop = max(stop, current_stop)
                    result = (current_stop - start, start, current_stop, result)
                    keywords[-1] = result
                    # whyyyyy ?? better commment it out
                    # else:
                    #     import pdb

                    #     pdb.set_trace()
                    #     current_stop = stop
                    #     result = (current_stop - start, start, current_stop, result)
                    #     keywords.append(result)
        return [x[3] for x in keywords]

    def extract(self, text):
        self.automaton.make_automaton()
        self.words_changed = False
        if self.replace_foreign_chars:
            text = unidecode.unidecode(text)
        keywords = []
        current_stop = -1
        _text = text.lower() if self._ignore_case_in_search else text
        for end_index, (length, result) in self.automaton.iter(_text):
            start = end_index - length + 1
            stop = end_index + 1
            # in:  text, start, stop, keyword
            # out: start, stop, keyword
            if self.http and result == "$HTTP":
                http_res = HTTP_FIX.search(_text[start:])
                if http_res:
                    stop = start + http_res.end()
                    res = (stop - start, start, stop, http_res.group())
                    if start > current_stop:
                        keywords.append(res)
                    elif stop - start > keywords[-1][0]:
                        keywords[-1] = res
                    else:
                        keywords.append(res)
                    current_stop = max(current_stop, stop)
                continue
            if start >= current_stop:
                result = self.verify_boundaries(
                    end_index, result, length, self.left_bound_chars, self.right_bound_chars, text
                )
                if result is None:
                    continue
                current_stop = stop
                keywords.append((stop - start, start, stop, result))
            elif stop - start > keywords[-1][0]:
                result = self.verify_boundaries(
                    end_index, result, length, self.left_bound_chars, self.right_bound_chars, text
                )
                if result is None:
                    continue
                current_stop = max(current_stop, stop)
                keywords[-1] = (stop - start, start, stop, result)
        return [x[-1] for x in keywords]

    def extract_overlapping(self, text):
        self.automaton.make_automaton()
        self.words_changed = False
        keywords = []
        if self.replace_foreign_chars:
            text = unidecode.unidecode(text)
        _text = text.lower() if self._ignore_case_in_search else text
        for end_index, (length, result) in self.automaton.iter(_text):
            if self.http and result == "$HTTP":
                start = end_index - length + 1
                http_res = HTTP_FIX.search(_text[start:])
                if http_res:
                    keywords.append(http_res.group())
                continue
            result = self.verify_boundaries(
                end_index, result, length, self.left_bound_chars, self.right_bound_chars, text
            )
            if result is None:
                continue
            keywords.append(result)
        return keywords

    def replace(self, text, return_entities=False):
        if self.returns != "norm":
            raise ValueError("no idea how i would do that")
        self.automaton.make_automaton()
        self.words_changed = False
        if self.replace_foreign_chars:
            text = unidecode.unidecode(text)
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
            text_ += text[stop1:start2] + result2[1]
        if return_entities:
            return text_, [x[-1] for x in keywords[1:-1]]
        return text_

    def _old_replace(self, text):
        if self.returns != "norm":
            raise ValueError("no idea how i would do that")
        self.automaton.make_automaton()
        self.words_changed = False
        if self.replace_foreign_chars:
            text = unidecode.unidecode(text)
        keywords = [(None, None, 0, "")]
        current_stop = -1
        _text = text.lower() if self._ignore_case_in_search else text
        for end_index, (length, result) in self.automaton.iter(_text):
            start = end_index - length + 1
            stop = end_index + 1
            if self.http and result == "$HTTP":
                http_res = HTTP_FIX.search(_text[start:])
                if http_res:
                    stop = start + http_res.end()
                    res = (stop - start, start, stop, http_res.group())
                    if start > current_stop:
                        keywords.append(res)
                    elif stop - start > keywords[-1][0]:
                        keywords[-1] = res
                    else:
                        keywords.append(res)
                    current_stop = max(current_stop, stop)
                continue
            if start >= current_stop:
                result = self.verify_boundaries(
                    end_index, result, length, self.left_bound_chars, self.right_bound_chars, text
                )
                if result is None:
                    continue
                current_stop = stop
                keywords.append((stop - start, start, stop, result))
            elif stop - start > keywords[-1][0]:
                result = self.verify_boundaries(
                    end_index, result, length, self.left_bound_chars, self.right_bound_chars, text
                )
                if result is None:
                    continue
                current_stop = max(current_stop, stop)
                keywords[-1] = (stop - start, start, stop, result)
        keywords.append((None, len(text), None, ""))
        text_ = ""
        for (_, start1, stop1, result1), (_, start2, stop2, result2) in zip(
            keywords[:-1], keywords[1:]
        ):
            text_ += text[stop1:start2] + result2
        return text_

    def verify_boundaries(
        self, end_index, result, length, left_non_allowed, right_non_allowed, text
    ):
        ind_after = end_index + 1
        try:
            if text[ind_after] in right_non_allowed:
                return None
        except IndexError:
            pass
        ind_before = end_index - length
        # watch out... cannot just get index as we might risk -1
        if ind_before != -1 and text[ind_before] in left_non_allowed:
            return None
        return self.extract_fn(ind_before + 1, ind_after, result, text)

    def extract_str(self, start_index, end, result, text):
        return result

    def extract_insensitive_match(self, start, end, result, text):
        match = text[start:end]
        return match

    def extract_insensitive_match_case(self, start, end, result, text):
        match = text[start:end]
        return match + "_" + determine_case(match)

    def extract_insensitive_norm_case(self, start, end, result, text):
        match = text[start, end]
        return result["norm"] + "_" + determine_case(match)

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
            elif self.returns == "match_case":
                extract_fn = self.extract_insensitive_match_case
            elif self.returns == "norm_case":
                extract_fn = self.extract_insensitive_norm_case
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
        if self._ignore_case_in_search or self.case == "smart":
            return key.lower() in self.automaton
        return key in self.automaton


# case sensitive vs insensitive
# complex return or simple return (str)
# complex gives atts, is_lower, is_upper, is_title, is_exact
# ability to add only title, lower, upper, and a shortcut: smart

# sought (case_sensitive=True|False, return_type=one_of_strats)
# found (start, end | length, value)
# normalized (reference to a normalized object with casing attributes)

# 8 return variations
# - just a normalized string
# - (is_lower, is_exact=True), (is_lower, is_exact=False)
# - (is_upper, is_exact=True), (is_upper, is_exact=False)
# - (is_title, is_exact=True), (is_title, is_exact=False)
# - (is_exact=True) (only is_exact means is_mixed)

# scikit
# - just text
# - hierarchical or not (text1_lower, text2_upper, text3_exact) vs (text1), (text2), (text3), (lower), (upper), (exact)

# case_return:
# - ignore_str
# - insensitive_str
# - smart_str
# - ignore_str
# - insensitive_str
# - smart_str
# - sensitive_simple
# - insensitive_complex
# - sensitive_complex
# - smart_complex

# user defines:
# case: ignore, insensitive, sensitive, smart
# returns: match, norm, match_case, norm_case, object


# ignore case, static (return the lower cased result regardless)
# insensitive matching, return string

# ignore case, dynamic (return the matched result found, this will be case sensitive)
# insensitive matching, return matched string

# ignore case, complex
# at "extraction time" fill the attributes
# # insensitive matching, and at extraction time come up with the atts

# when smart (sensitive) and complex
# at "add time" fill the attributes for the variations on the inputs

# when sensitive and complex
# at "extraction time" fill the attributes
# # do actually insensitive matching, and at extraction time come up with the atts


# ts = TextSearch("insensitive", "match")
# ts.add("hi")
# ts.tokenized_extract("hi")

# ts = ts.from_ts(ts)
# ts.tokenized_extract("hi")

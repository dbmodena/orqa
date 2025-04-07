from functools import lru_cache
import inflection
import unicodedata


@lru_cache(maxsize=int(1e7))
def is_num(x):
    "Very very simple solution, but for many cases it's fine"
    if isinstance(x, str):
        # chars '_' and '/' are kept, they are used often
        # for datetime, like 2021_2023 which may be more
        # interesting than basic numbers
        x = x.replace(',', '.').replace('%', ' ')
    try: 
        float(x)
    except: 
        return False
    return True


characters_translator = str.maketrans("\n ,.\"", "_____")

@lru_cache(maxsize=int(1e7))
def sanitize_string(s):
    """
    Replaces problematic characters in column names with underscores,
    normalizes accents, and strips spaces.
    """

    # inflection
    s = inflection.underscore(str(s)).lower()
    # normalize accents (e.g., é -> e)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    # replace problematic characters with underscores
    return s.translate(characters_translator).strip()
            




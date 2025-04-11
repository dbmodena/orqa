from functools import lru_cache
import re
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


replace_chars = "\n \\\"()[]"
characters_translator = str.maketrans(replace_chars, "_" * len(replace_chars))

@lru_cache(maxsize=int(1e7))
def sanitize_string(s):
    """
    Replaces problematic characters in column names with underscores,
    normalizes accents, and strips spaces.
    """

    # inflection
    if isinstance(s, str):
        s = inflection.underscore(s).lower()
    else:
        s = str(s)
    # normalize accents (e.g., é -> e)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    # replace problematic characters with underscores
    return s.translate(characters_translator).strip()



def get_resource_metadata(rsc_id, table_ids, metadata):
    # the pure resource ID should be the one without the underscore _#value 
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    rsc = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    
    # get metadata and tags if present
    pkg_keywords = []
    if 'keywords' in metadata[rsc_id] and 'en' in metadata[rsc_id]['keywords']:
        pkg_keywords = metadata[rsc_id]['keywords']['en']

    pkg_tags = []
    if 'tags' in metadata[rsc_id]:
        pkg_tags = metadata[rsc_id]['tags']

    pkg_id = metadata[rsc_id]['id']
    pkg_title = metadata[rsc_id]['title']
    pkg_notes = metadata[rsc_id]['notes']
    rsc_name = rsc['name']

    return rsc_id, rsc_name, pkg_id, pkg_title, pkg_notes, pkg_keywords, pkg_tags





from typing import List


class RawTokenSet:
    def __init__(self, set_id: int, tokens: List[int], raw_tokens: List[bytes]):
        self.set_id = set_id
        self.tokens = tokens
        self.raw_tokens = raw_tokens


class ListEntry:
    def __init__(self, set_id: int, size: int, match_position: int):
        self.set_id = set_id
        self.size = size
        self.match_position = match_position


class BySize:
    def __init__(self, entries: List[ListEntry]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    def __setitem__(self, index: int, value: ListEntry):
        self.entries[index] = value

    def swap(self, i: int, j: int):
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def less(self, i: int, j: int) -> bool:
        return self.entries[i].size < self.entries[j].size


class ByPrefixLength:
    def __init__(self, entries: List[ListEntry]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    def __setitem__(self, index: int, value: ListEntry):
        self.entries[index] = value

    def swap(self, i: int, j: int):
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def less(self, i: int, j: int) -> bool:
        return self.entries[i].match_position > self.entries[j].match_position


class BySuffixLength:
    def __init__(self, entries: List[ListEntry]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    def __setitem__(self, index: int, value: ListEntry):
        self.entries[index] = value

    def swap(self, i: int, j: int):
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def less(self, i: int, j: int) -> bool:
        return (self.entries[i].size - self.entries[i].match_position) > (self.entries[j].size - self.entries[j].match_position)


# def set_tokens(db, table: str, set_id: int) -> List[int]:
#     query = f"SELECT tokens FROM {table} WHERE id = %s;"
#     with db.cursor() as cursor:
#         cursor.execute(query, (set_id,))
#         tokens = cursor.fetchone()[0]
#     return tokens


# def set_tokens_prefix(db, table: str, set_id: int, end_pos: int) -> List[int]:
#     query = f"SELECT tokens[1:%s] FROM {table} WHERE id = %s;"
#     with db.cursor() as cursor:
#         cursor.execute(query, (end_pos + 1, set_id))
#         tokens = cursor.fetchone()[0]
#     return tokens


# def set_tokens_suffix(db, table: str, set_id: int, start_pos: int) -> List[int]:
#     query = f"SELECT tokens[%s:size] FROM {table} WHERE id = %s;"
#     with db.cursor() as cursor:
#         cursor.execute(query, (start_pos + 1, set_id))
#         tokens = cursor.fetchone()[0]
#     return tokens


# def set_tokens_subset(db, table: str, set_id: int, start_pos: int, end_pos: int) -> List[int]:
#     query = f"SELECT tokens[%s:%s] FROM {table} WHERE id = %s;"
#     with db.cursor() as cursor:
#         cursor.execute(query, (start_pos + 1, end_pos, set_id))
#         tokens = cursor.fetchone()[0]
#     return tokens

# 
# def inverted_list(token: int) -> List[ListEntry]:
#     # query = f"SELECT set_ids, set_sizes, match_positions FROM {table} WHERE token = %s"
#     # with db.cursor() as cursor:
#     #     cursor.execute(query, (token,))
#     #     set_ids, sizes, match_positions = cursor.fetchone()
#     set_ids, sizes, match_positions = db.get_inverted_list(token).fetchone()
#     entries = []
#     for i in range(len(set_ids)):
#         entry = ListEntry(
#             set_id=set_ids[i],
#             size=int(sizes[i]),
#             match_position=int(match_positions[i])
#         )
#         entries.append(entry)
#     return entries
# 
# 
# def query_sets(db:JOSIEDBHandler) -> List[RawTokenSet]:
#     # query = f"""
#     #     SELECT id, (
#     #         SELECT array_agg(raw_token)
#     #         FROM {list_table}
#     #         WHERE token = any(tokens)
#     #     ), tokens
#     #     FROM {query_table};
#     # """
#     # with db.cursor() as cursor:
#     #     cursor.execute(query)
#     #     rows = cursor.fetchall()
#     
#     queries = []
#     # for row in rows:
#     for row in db.get_query_sets():
#         set_id, raw_tokens, tokens = row
#         query = RawTokenSet(set_id, tokens, raw_tokens)
#         queries.append(query)
#     
#     return queries

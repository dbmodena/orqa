import hashlib
import logging
from typing import List, Tuple

from orqa.josie.db import JOSIEDBHandler
from orqa.josie.io import RawTokenSet


# Constants
MINHASH_SEED = 42
MINHASH_SIZE = 200

MIN_READ_COST = 1000000.0
READ_SET_COST_SLOPE = 1253.19054300781
READ_SET_COST_INTERCEPT = -9423326.99507381
READ_LIST_COST_SLOPE = 1661.93366983753
READ_LIST_COST_INTERCEPT = 1007857.48225696


class TokenMapEntry:
    def __init__(self, token: int, group_id: int):
        self.token = token
        self.group_id = group_id


class TokenTable:
    def process(self, raw_token_set) -> Tuple[List[int], List[int], List[int]]:
        raise NotImplementedError

    def process_and_minhash_signature(self, raw_token_set) -> Tuple[List[int], List[int]]:
        raise NotImplementedError


class TokenTableMem(TokenTable):
    def __init__(self, db:JOSIEDBHandler, ignore_self: bool):
        self.db = db
        self.ignore_self = ignore_self
        self.token_map = {}
        self.frequencies = []
        self._initialize_token_map()

    def _initialize_token_map(self):
        logging.info("Initializing token map...")
        # Fetch the number of entries
        count = self.db.count_posting_lists()

        self.token_map = {}
        self.frequencies = []

        logging.info(f"Initialized token map, {count} entries")

        # Find max duplicate group id
        max_gid = self.db.max_duplicate_group_id()
        self.frequencies = [0] * (max_gid + 1)

        logging.info(f"Initialized frequency table, {len(self.frequencies)} entries")

        # Load all tokens and duplicate group ids
        logging.info("Filling token table entries...")
        count = 0
        for row in self.db.posting_lists__memproc():
            raw_token, token, frequency, group_id = row
            # TODO hashlib doen't support fnv1a with 64 bits,
            # it this a significative change?
            # h = hashlib.new('fnv1a_64')
            h = hashlib.sha256()
            h.update(raw_token)
            hash_value = h.digest()

            # Assign frequency
            self.frequencies[group_id] = frequency
            # Assign token entry
            self.token_map[hash_value] = TokenMapEntry(token, group_id)
            count += 1
            if count % 1000 == 0:
                print(f"\r{count} read", end="")
        print()
        logging.info("Finished creating token map and frequency table")

    def process(self, raw_token_set:RawTokenSet) -> Tuple[List[int], List[int], List[int]]:
        tokens = []
        counts = []
        gids = []

        for raw_token in raw_token_set.raw_tokens:
            h = hashlib.sha256(raw_token) # hashlib.new('fnv1a_64')
            hash_value = h.digest()
            if hash_value in self.token_map:
                entry = self.token_map[hash_value]
                frequency = self.frequencies[entry.group_id]
                if self.ignore_self and frequency < 2:
                    continue
                tokens.append(entry.token)
                counts.append(frequency - 1)
                gids.append(entry.group_id)

        sorted_indices = sorted(range(len(tokens)), key=lambda i: tokens[i])
        tokens = [tokens[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        gids = [gids[i] for i in sorted_indices]

        return tokens, counts, gids
    

class TokenTableDisk(TokenTable):
    def __init__(self, db:JOSIEDBHandler, ignore_self: bool):
        self.db = db
        self.ignore_self = ignore_self

    def process(self, raw_token_set) -> Tuple[List[int], List[int], List[int]]:
        tokens = []
        counts = []
        gids = []

        for row in self.db.posting_lists__diskproc(raw_token_set, self.ignore_self):
            token, count, gid = row
            tokens.append(token)
            counts.append(count)
            gids.append(gid)

        return tokens, counts, gids

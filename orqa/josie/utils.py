from typing import Any, List

from orqa.josie.common import next_distinct_list
from orqa.josie.db import JOSIEDBHandler


class CandidateEntry:
    def __init__(self, id, size, candidate_current_position, query_current_position, skipped_overlap):
        self.id = id
        self.size = size
        self.first_match_position = candidate_current_position
        self.latest_match_position = candidate_current_position
        self.query_first_match_position = query_current_position
        self.partial_overlap = skipped_overlap + 1  # overlapping token skipped + the token at the current position
        self.maximum_overlap = 0
        self.estimated_overlap = 0
        self.estimated_cost = 0.0
        self.estimated_next_upperbound = 0
        self.estimated_next_truncation = 0
        self.read = False

    def update(self, candidate_current_position, skipped_overlap):
        self.latest_match_position = candidate_current_position
        self.partial_overlap += skipped_overlap + 1  # skipped + this position

    def upperbound_overlap(self, query_size, query_current_position):
        self.maximum_overlap = self.partial_overlap + min(query_size - query_current_position - 1, self.size - self.latest_match_position - 1)
        return self.maximum_overlap

    def est_overlap(self, query_size, query_current_position):
        self.estimated_overlap = int((self.partial_overlap / (query_current_position + 1 - self.query_first_match_position)) * (query_size - self.query_first_match_position))
        self.estimated_overlap = min(self.estimated_overlap, self.upperbound_overlap(query_size, query_current_position))
        return self.estimated_overlap

    def est_cost(self, db:JOSIEDBHandler):
        self.estimated_cost = db.read_set_cost(self.suffix_length())
        return self.estimated_cost

    def est_truncation(self, query_size, query_current_position, query_next_position):
        self.estimated_next_truncation = int(((query_next_position - query_current_position) / (query_size - self.query_first_match_position)) * (self.size - self.first_match_position))
        return self.estimated_next_truncation

    def est_next_overlap_upperbound(self, query_size, query_current_position, query_next_position):
        query_jump_length = query_next_position - query_current_position
        query_prefix_length = query_current_position + 1 - self.query_first_match_position
        additional_overlap = int((self.partial_overlap / query_prefix_length) * query_jump_length)
        next_latest_matching_position = int((query_jump_length / (query_size - self.query_first_match_position)) * (self.size - self.first_match_position)) + self.latest_match_position
        self.estimated_next_upperbound = self.partial_overlap + additional_overlap + min(query_size - query_next_position - 1, self.size - next_latest_matching_position - 1)
        return self.estimated_next_upperbound

    def suffix_length(self):
        return self.size - self.latest_match_position - 1

    def check_min_sample_size(self, query_current_position, batch_size):
        return (query_current_position - self.query_first_match_position + 1) > batch_size


class ByEstimatedOverlap:
    def __init__(self, candidates):
        self.candidates = candidates

    def __lt__(self, other):
        if self.candidates.estimated_overlap == other.candidates.estimated_overlap:
            return self.candidates.estimated_cost < other.candidates.estimated_cost
        return self.candidates.estimated_overlap > other.candidates.estimated_overlap


class ByMaximumOverlap:
    def __init__(self, candidates):
        self.candidates = candidates

    def __lt__(self, other):
        return self.candidates.maximum_overlap < other.candidates.maximum_overlap


class ByFutureMaxOverlap:
    def __init__(self, candidates):
        self.candidates = candidates

    def __lt__(self, other):
        return self.candidates.estimated_next_upperbound < other.candidates.estimated_next_upperbound


def upperbound_overlap_unknown_candidate(query_size, query_current_position, prefix_overlap):
    return query_size - query_current_position + prefix_overlap


def next_batch_distinct_lists(tokens, gids, curr_index, batch_size):
    n = 0
    next_index, _ = next_distinct_list(tokens, gids, curr_index)
    while next_index < len(tokens):
        curr_index = next_index
        n += 1
        if n == batch_size:
            break
        next_index, _ = next_distinct_list(tokens, gids, curr_index)
    return curr_index


def prefix_length(query_size, kth_overlap):
    if kth_overlap == 0:
        return query_size
    return query_size - kth_overlap + 1


def read_lists_benefit_for_candidate(db:JOSIEDBHandler, ce:CandidateEntry, kth_overlap):
    if kth_overlap >= ce.estimated_next_upperbound:
        return ce.estimated_cost
    return ce.estimated_cost - db.read_set_cost(ce.suffix_length() - ce.estimated_next_truncation)


def process_candidates_init(db:JOSIEDBHandler, query_size, query_current_position, next_batch_end_index,
                            kth_overlap, min_sample_size, candidates:dict[int,CandidateEntry], ignores) -> tuple[float|Any, int, List[CandidateEntry]] :
    read_lists_benefit = 0.0
    qualified = []
    num_with_benefit = 0

    for ce in list(candidates.values()):
        ce.upperbound_overlap(query_size, query_current_position)
        if kth_overlap >= ce.maximum_overlap:
            del candidates[ce.id]
            ignores[ce.id] = True
            continue

        if not ce.check_min_sample_size(query_current_position, min_sample_size):
            continue

        ce.est_cost(db)
        ce.est_overlap(query_size, query_current_position)
        ce.est_truncation(query_size, query_current_position, next_batch_end_index)
        ce.est_next_overlap_upperbound(query_size, query_current_position, next_batch_end_index)

        read_lists_benefit += read_lists_benefit_for_candidate(db, ce, kth_overlap)

        qualified.append(ce)

        if ce.estimated_overlap > kth_overlap:
            num_with_benefit += 1

    return read_lists_benefit, num_with_benefit, qualified


def process_candidates_update(db, kth_overlap, candidates, counter, ignores):
    read_lists_benefit = 0.0

    for j, ce in enumerate(candidates):
        if ce is None or ce.read:
            continue
        if ce.maximum_overlap <= kth_overlap:
            candidates[j] = None
            del counter[ce.id]
            ignores[ce.id] = True
        read_lists_benefit += read_lists_benefit_for_candidate(db, ce, kth_overlap)

    return read_lists_benefit


def read_set_benefit(query_size, kth_overlap, kth_overlap_after_push, candidates, read_list_costs, fast):
    b = 0.0
    if kth_overlap_after_push <= kth_overlap:
        return b

    p0 = prefix_length(query_size, kth_overlap)
    p1 = prefix_length(query_size, kth_overlap_after_push)

    b += read_list_costs[p0 - 1] - read_list_costs[p1 - 1]

    if fast:
        return b

    for ce in candidates:
        if ce is None or ce.read:
            continue
        if ce.maximum_overlap <= kth_overlap_after_push:
            b += ce.estimated_cost

    return b

import time

from orqa.josie.db import JOSIEDBHandler
from orqa.josie.io import RawTokenSet
from orqa.josie.tokentable import TokenTable
from orqa.josie.common import overlap
from orqa.josie.exp import ExperimentResult, write_result_string

from orqa.josie.heap import (
    SearchResultHeap, 
    push_candidate, 
    kth_overlap, 
    kth_overlap_after_push, 
    ordered_results
)

from orqa.josie.utils import (
    CandidateEntry, 
    process_candidates_init,
    process_candidates_update,
    upperbound_overlap_unknown_candidate, 
    next_batch_distinct_lists, 
    read_set_benefit,
    read_lists_benefit_for_candidate
)


def josie_core(
    db: JOSIEDBHandler,
    tb: TokenTable,
    query: RawTokenSet,
    k: int,
    batch_size: int = 20,
    expensive_estimation_budget: int = 5000,
    ignore_self: bool = True
):
    exp_result = ExperimentResult()

    start = time.time()
    tokens, freqs, gids = tb.process(query)
    read_list_costs = [0.0] * len(freqs)
    for i in range(len(freqs)):
        if i == 0:
            read_list_costs[i] = db.read_list_cost(freqs[i] + 1)
        else:
            read_list_costs[i] = read_list_costs[i - 1] + db.read_list_cost(freqs[i] + 1)

    exp_result.preproc_duration = int((time.time() - start) * 1000)  # in milliseconds
    start = time.time()

    query_size = len(tokens)
    counter = {}
    ignores = {query.set_id: True} if ignore_self else {}
    h = SearchResultHeap()
    num_skipped = 0

    curr_batch_lists = batch_size

    for i in range(query_size):
        token = tokens[i]
        skipped_overlap = num_skipped
        max_overlap_unseen_candidate = upperbound_overlap_unknown_candidate(query_size, i, skipped_overlap)

		# Early terminates once the threshold index has reached and
		# there is no remaining sets in the counter
        if kth_overlap(h, k) >= max_overlap_unseen_candidate and len(counter) == 0:
            break
        
        # Read the list
        entries = db.get_inverted_list(token)
        exp_result.num_list_read += 1
        exp_result.max_list_size_read = max(exp_result.max_list_size_read, len(entries))

        # Merge this list and compute counter entries
        # Skip sets that has been computed for exact overlap previously
        for entry in entries:
            if entry.set_id in ignores:
                continue
            
            # Process seen candidates
            if entry.set_id in counter:
                ce = counter[entry.set_id]
                ce.update(entry.match_position, skipped_overlap)
                continue
            
            # No need to process unseen candidate if we have reached this point
            if kth_overlap(h, k) >= max_overlap_unseen_candidate:
                continue
            counter[entry.set_id] = CandidateEntry(entry.set_id, entry.size, entry.match_position, i, skipped_overlap)

        # Terminates as we are at the last list, no need to read set
        if i == query_size - 1:
            break

        # Continue reading the next list when there is no candidates
        # OR do not start reading sets until we have seen at least k candidates
        # Continue reading the next list when we are still in the current batch
        if len(counter) == 0 or (len(counter) < k and len(h) < k) or curr_batch_lists > 0:
            curr_batch_lists -= 1
            continue

        # Reset counter
        curr_batch_lists = batch_size

        # Find the end index of the next batch of posting lists
        next_batch_end_index = next_batch_distinct_lists(tokens, gids, i, batch_size)

        # Compute the cost of reading the next batch of posting lists
        merge_lists_cost = read_list_costs[next_batch_end_index] - read_list_costs[i]

        # Process candidates to estimate benefit of reading the next batch of posting lists
        # and obtain qualified candidates
        merge_lists_benefit, num_with_benefit, candidates = process_candidates_init(
            db, query_size, i, next_batch_end_index, kth_overlap(h, k), batch_size, counter, ignores
        )

        # Record the counter size
        exp_result.max_counter_size = max(exp_result.max_counter_size, len(counter))

        # Continue reading posting lists if no qualified cadidate found
        # or no candidates can bring positive benefit
        if num_with_benefit == 0 or len(candidates) == 0:
            continue

        # Sort candidates by estimated overlaps
        candidates.sort(key=lambda c: int(c.estimated_overlap), reverse=True)
        
        # Keep track of the estimation budget
        prev_kth_overlap = kth_overlap(h, k)
        num_candidate_expensive = 0
        fast_estimate = False
        
        # the kth overlap used for fast estimation
        fast_estimate_kth_overlap = 0

        # Greedily determine the next best candidate until the qualified
		# candidates exhausted or when reading the next batch of lists yield
		# better net benefit
        for candidate in candidates:
            # Skip ones that are already been eliminated
            if candidate is None:
                continue

            # The current kth overlap before reading the current candidate
            kth = kth_overlap(h, k)
            
            # Stop when the current candidate is no longer 
            # expected to bring positive effect
            if candidate.estimated_overlap <= kth:
                break

            # Always read candidate when we have not had running top-k yet
            if len(h) >= k:
                # Increase the number of candidates that has used expensive benefit estimation
                num_candidate_expensive += 1

                # Switch to fast estimate if estimation budget has reached
                if not fast_estimate and num_candidate_expensive * len(candidates) > expensive_estimation_budget:
                    fast_estimate = True
                    fast_estimate_kth_overlap = prev_kth_overlap
                if not fast_estimate:
                    merge_lists_benefit = process_candidates_update(db, kth, candidates, counter, ignores)

                # Estimate the benefit of reading this set
				# (expensive if fastEstimate is false)
                probe_set_benefit = read_set_benefit(
                    query_size, kth, kth_overlap_after_push(h, k, candidate.estimated_overlap),
                    candidates, read_list_costs, fast_estimate)
                probe_set_cost = candidate.estimated_cost

                # Stop looking at candidates if the current best one is no
				# better than reading the next batch of posting lists
				# The next best one either has lower benefit, which is
				# monotonic w.r.t. the overlap, or higher cost.
				# So if the current best one is not
				# better the next best one will be even worse.
                if probe_set_benefit - probe_set_cost < merge_lists_benefit - merge_lists_cost:
                    break

            # Now read this candidate
            # Decrease merge list benefit if we are using fast estimate
            if fast_estimate or (num_candidate_expensive + 1) * len(candidates) > expensive_estimation_budget:
                merge_lists_benefit -= read_lists_benefit_for_candidate(db, candidate, fast_estimate_kth_overlap)

            candidate.read = True
            ignores[candidate.id] = True
            del counter[candidate.id]

            # We are done if this candidate can be pruned, this can happen
            # sometimes when using fast estimate
            if candidate.maximum_overlap <= kth:
                continue
            
            # Compute the total overlap
            total_overlap = 0
            if candidate.suffix_length() > 0:
                s = db.get_set_tokens_by_suffix(candidate.id, candidate.latest_match_position + 1)
                exp_result.num_set_read += 1
                exp_result.max_set_size_read = max(exp_result.max_set_size_read, len(s))
                suffix_overlap = overlap(s, tokens[i + 1:])
                total_overlap = suffix_overlap + candidate.partial_overlap
            else:
                total_overlap = candidate.partial_overlap

            # Save the current kth overlap as the previous kth overlap
            prev_kth_overlap = kth
            # Push the candidate to the heap
            push_candidate(h, k, candidate.id, total_overlap)

    for ce in counter.values():
        push_candidate(h, k, ce.id, ce.partial_overlap)

    results = ordered_results(h)

    exp_result.duration = int((time.time() - start) * 1000)  # in milliseconds
    exp_result.results = write_result_string(results)
    exp_result.query_id = query.set_id
    exp_result.query_size = len(query.raw_tokens)
    exp_result.num_result = len(results)
    exp_result.ignore_size = len(ignores)
    exp_result.query_num_token = len(tokens)
    
    return results, exp_result

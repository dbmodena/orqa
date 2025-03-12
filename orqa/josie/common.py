import math


# This variable corresponds to totalNumberOfSets in the Go code.
total_number_of_sets = 1.0

# Initialize function equivalent to Go's init function
def init():
    global total_number_of_sets
    # Optional setup: total_number_of_sets can be set here if desired.
    total_number_of_sets = 1.0

# Pruning power upper bound function
def pruning_power_ub(freq, k):
    # Math.Log in Go is equivalent to math.log in Python
    return math.log((min(k, freq) + 0.5) * (total_number_of_sets - k - freq + min(k, freq) + 0.5) /
                     ((max(0, k - freq) + 0.5) * (max(freq - k, 0) + 0.5)))

# Inverse set frequency function
def inverse_set_frequency(freq):
    return math.log(total_number_of_sets / float(freq))

# Function to find the next distinct list index and the number of skipped elements
def next_distinct_list(tokens, gids, curr_list_index):
    if curr_list_index == len(tokens) - 1:
        return len(tokens), 0
    
    num_skipped = 0
    for i in range(curr_list_index + 1, len(tokens)):
        if i < len(tokens) - 1 and gids[i + 1] == gids[i]:
            num_skipped += 1
            continue
        list_index = i
        break
    
    return list_index, num_skipped

# Function to calculate overlap between two lists
def overlap(set_tokens, query_tokens):
    i, j = 0, 0
    overlap = 0
    while i < len(query_tokens) and j < len(set_tokens):
        d = query_tokens[i] - set_tokens[j]
        if d == 0:
            overlap += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return overlap

# Function to calculate overlap and update counts
def overlap_and_update_counts(set_tokens, query_tokens, counts):
    i, j = 0, 0
    overlap = 0
    while i < len(query_tokens) and j < len(set_tokens):
        d = query_tokens[i] - set_tokens[j]
        if d == 0:
            counts[i] -= 1
            overlap += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return overlap

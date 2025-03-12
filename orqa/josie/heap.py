from typing import List

import heapq


class SearchResult:
    def __init__(self, id: int, overlap: int):
        self.id = id
        self.overlap = overlap

    def __lt__(self, other):
        return self.overlap < other.overlap  # This is the comparison function used by heapq


class SearchResultHeap:
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)
    
    def push(self, result: SearchResult):
        heapq.heappush(self.heap, result)

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        if self.heap:
            return self.heap[0]
        return None


def kth_overlap(heap: SearchResultHeap, k: int) -> int:
    if len(heap) < k:
        return 0
    return heap.peek().overlap


def push_candidate(heap: SearchResultHeap, k: int, id: int, overlap: int) -> bool:
    if len(heap) == k:
        if heap.peek().overlap >= overlap:
            return False
        heap.pop()
    heap.push(SearchResult(id, overlap))
    return True


def ordered_results(heap: SearchResultHeap) -> List[SearchResult]:
    result = []
    while len(heap) > 0:
        result.append(heap.pop())
    return result[::-1]  # Return in reverse order (top-k first)


def kth_overlap_after_push(heap: SearchResultHeap, k: int, overlap: int) -> int:
    if len(heap) < k - 1:
        return 0
    kth = heap.peek().overlap
    if overlap <= kth:
        return kth
    if k == 1:
        return overlap
    jth = heap.heap[1].overlap if k == 2 else min(heap.heap[1].overlap, heap.heap[2].overlap)
    return min(jth, overlap)


def copy_heap(heap: SearchResultHeap) -> SearchResultHeap:
    new_heap = SearchResultHeap()
    new_heap.heap = heap.heap[:]
    heapq.heapify(new_heap.heap)  # Re-heapify after copying
    return new_heap

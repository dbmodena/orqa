import csv
from typing import Optional
from dataclasses import dataclass

from orqa.josie.heap import SearchResult


@dataclass
class ExperimentResult:
    # Query details
    query_id: int = 0
    query_size: int = 0
    query_num_token: int = 0
    num_result: int = 0
    duration: int = 0
    preproc_duration: int = 0
    num_set_read: int = 0
    num_list_read: int = 0
    num_byte_read: int = 0
    max_set_size_read: int = 0
    max_list_size_read: int = 0
    max_counter_size: int = 0
    ignore_size: int = 0
    actions: str = ""  # Example: "lso" (list, set, overlap)
    results: str = ""  # Example: "so" (set, overlap)

    # These properties are for the merge probe algorithm only
    benefit_costs: Optional[str] = None

    # These properties are for the LSH Ensemble algorithm only
    lsh_duration: Optional[int] = None
    lsh_precision: Optional[float] = None

    # Method to convert the object to a dictionary for CSV writing
    def to_dict(self):
        return {
            "query_id": self.query_id,
            "query_size": self.query_size,
            "query_num_token": self.query_num_token,
            "num_result": self.num_result,
            "duration": self.duration,
            "preproc_duration": self.preproc_duration,
            "num_set_read": self.num_set_read,
            "num_list_read": self.num_list_read,
            "num_byte_read": self.num_byte_read,
            "max_set_size_read": self.max_set_size_read,
            "max_list_size_read": self.max_list_size_read,
            "max_counter_size": self.max_counter_size,
            "ignore_size": self.ignore_size,
            "actions": self.actions,
            "results": self.results,
            "benefit_cost": self.benefit_costs 
        }

    # Method to write this object to a CSV file
    @staticmethod
    def write_to_csv(filename, data):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].to_dict().keys())
            writer.writeheader()
            for row in data:
                writer.writerow(row.to_dict())


def write_result_string(results: list[SearchResult]) -> str:
    s = ""
    for result in results:
        s += f"s{result.id}o{result.overlap}"
    return s

def write_all_results(results: list[ExperimentResult], filename: str):
    with open(filename, 'w') as fw:
        fw.write(','.join(results[0].to_dict().keys()) + '\n')
        for r in results:
            fw.write(','.join(map(str, r.to_dict().values())) + '\n')



"""
Candidates Discovery Stage

In this stage, candidates for actual dataset generation in next steps
are proposed through an agentic step.

1 - The CandidatesDiscoveryAgent agent analyses a random subset of datasets
    drawned from the whole available collection;
        a. for each dataset, it inspects its available metadata and a sample
            of few rows;
        b. if the dataset is not recognized as valid, any

"""

from ..conf import OrQAConfig

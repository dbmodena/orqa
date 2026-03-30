The solr.py is a simple script to handle Solr clusters creation/deletion with Python Path helper class and functions.


### Prerequisites
Install Java (tested on OpenJDK 21.0.10).

Install Solr (tested on 9.10.0): https://solr.apache.org/downloads.html

Define a env variable DATA_DIR pointing to the location where OrQA data are stored.

Define a env variable SOLR_PATH pointing to the root folder where Solr is installed.


### Use the script

Run:

```sh
python3 solr create {uk,nyc}
```

to create the desired cluster with pre-defined configuration.

Navigate to 
`http://localhost:8983/solr`

to see and explore the just created cluster. Now, the server should be up and metadata uploaded and indexed correctly for the given configuration.

Run:

```sh
python3 solr delete {uk,nyc}
```

to delete an existing cluster. See --help for other operations (but it's just a wrapper around the solr CLI tool).






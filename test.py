from io import BytesIO
import certifi
import pandas as pd
import urllib3

# url = 'https://catalogue.data.gov.bc.ca/dataset/2b44d212-5438-47a9-ad23-20eb8ada9709/resource/c7cc9297-220c-4d6c-a9a7-72d0680b2f74/download/service-bc-locations-update-05-08-20.csv'
url = 'https://www150.statcan.gc.ca/n1/tbl/csv/32100142-eng.zip'

http = urllib3.PoolManager(
    retries=False,
    timeout=urllib3.Timeout(connect=7.0, read=5.0),        
    headers={
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0'
    },
    cert_file=certifi.where(),
    ca_certs=certifi.where(),
    cert_reqs="CERT_REQUIRED"
)

response = http.request('GET', url)
pd_read_csv_kwargs = {'sep': None, 'encoding': 'latin-1', 'encoding_errors': 'ignore', 'on_bad_lines': 'skip', 'engine': 'python'}

print(response.status)
# df = pd.read_csv(BytesIO(response.data), **pd_read_csv_kwargs)
# print(df)

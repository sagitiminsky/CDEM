import json
import requests
import pandas as pd
import CellEnMon.config as config
import os
import ast
#from google.cloud import storage
# from CellEnMon.libs.vault.vault import VaultService
import numpy as np 
# vault_service=VaultService()
url = "https://api.ims.gov.il/v1/envista/stations/64/data?from=2019/12/01&to=2020/01/01"

headers = {
    'Authorization': 'ApiToken ' + "f058958a-d8bd-47cc-95d7-7ecf98610e47"
}

## Setting credentials using the downloaded JSON file
path = 'CellEnMon/cellenmon-e840a9ba53e8.json'
SELECTOR = ['DOWNLOAD']  # DOWNLOAD

# UPLOAD IS DONE VIA DATASET MANAGER

# if "UPLOAD" in SELECTOR:
#     client = storage.Client.from_service_account_json(json_credentials_path=path)
#     ## For slow upload speed
#     storage.blob._DEFAULT_CHUNKSIZE = 2097152  # 1024 * 1024 B * 2 = 2 MB
#     storage.blob._MAX_MULTIPART_SIZE = 2097152  # 2 MB


class IMS_Scrapper_obj:
    def __init__(self, index, station_id, station_name, location, _from, _to):
        self.index = index
        self.station_id = station_id
        self._from = _from
        self._to = _to
        self.root = config.ims_root_files
        self.station_id = station_id
        self.station_name = station_name
        self.station_location = location
        self.station_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}/data/?from={_from}&to={_to}"
        if "UPLOAD" in SELECTOR:
            self.bucket = client.get_bucket('cell_en_mon')

    # def upload_files_to_gcs(self):
    #     for station in os.listdir(self.root):
    #         blob = self.bucket.blob(
    #             f'ims/{config.start_date_str_rep_ddmmyyyy}-{config.end_date_str_rep_ddmmyyyy}/raw/{station}')
    #         try:
    #             with open(f"{self.root}/raw/{station}", 'rb') as f:
    #                 blob.upload_from_file(f)
    #             print(f'Uploaded file:{station} succesfully !')
    #         except Exception as e:
    #             print(f'Uploaded file:{station} failed with the following exception:{e}!')

    def get_entry(self, arr, type):
        i = 0
        while (not ('name' in arr[i] and arr[i]['name'] == type)):
            i += 1

        return arr[i]

    def download_from_ims(self):
        data_response = requests.request("GET", self.station_data, headers=headers)

        if data_response.status_code != 200:
            print("station id: {} , data response: {}".format(self.station_id, data_response.status_code))

        elif self.station_location['latitude'] and self.station_location['longitude']:
            try:
                data = json.loads(data_response.text.encode('utf8'))
                
                file_name = "{}_{}_{}.csv".format(self.station_name, self.station_location['longitude'], self.station_location['latitude'])
                if not os.path.exists(self.root):
                    os.makedirs(f'{self.root}/raw')
                
                
                df=pd.DataFrame(data['data'])
                ims_vec = np.array([])
                time = np.array([" ".join(t.split('+')[0].split('T')) for t in df.datetime])
                if df is not None:
                    for row in list(df.channels):
                        ims_vec = np.append(ims_vec, np.array([row[0]['value']]))

                
                    data = {'Time': time, 'RainAmout[mm/h]': ims_vec}
                    pd.DataFrame.from_dict(data).to_csv(f'{self.root}/raw/{file_name}', index=False)
                
            except (json.decoder.JSONDecodeError):
                print("station id: {} , data response: {}".format(self.station_id, "JSONDecodeError"))


#
if __name__ == "__main__":
    for index, station in enumerate(config.ims_mapping):
        print('processing station: {}/{}'.format(index+1, config.ims_scrape_config['total_number_of_ims_stations']))
        obj = IMS_Scrapper_obj(index=index, station_id=station['stationId'], station_name=station['name'],
                               location=station['location'],
                               _from=config.ims_scrape_config['_from'], _to=config.ims_scrape_config['_to'])

        if 'DOWNLOAD' in SELECTOR:
            obj.download_from_ims()

        # if 'UPLOAD' in SELECTOR:
        #     obj.upload_files_to_gcs()

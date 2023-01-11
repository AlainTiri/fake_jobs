#
# from google.cloud import bigquery
# from google.oauth2 import service_account
#
#
class Config:
    PROJECT_ID = 'fake_project'
    OPTINS_DATASET = "input"
    OPTINS_TABLE = "bigquery"
    SERVICE_ACCOUNT_FILE = "../config/defaultserviceaccount.json"


class ToBigQuery:
    # def __init__(self):
    #     self.config = Config()
    #     cred = service_account.Credentials.from_service_account_file(Config.SERVICE_ACCOUNT_FILE)
    #     self.client = bigquery.Client(project=self.PROJECT_ID, credentials=cred)
    #     self.schema = [
    #         {
    #
    #         }
    #     ]

    def sendToBQ(self, description, prediction):
        return "uploading."

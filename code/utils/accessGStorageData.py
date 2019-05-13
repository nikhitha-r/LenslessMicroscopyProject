from google.cloud import exceptions
from google.auth.transport.requests import AuthorizedSession

from google.cloud import storage



SCOPES=['https://www.googleapis.com/auth/devstorage.read_write']


class StorageAccess(object):
    """ with proper key-files provides API to access data on cloud storage
    """

    def __init__(self,pathToFiles):
        self.project_id = 'tumDIlabProject86419'
		self.SERVICE_ACCOUNT_FILE = 'PUT_serviceaccountkeys_IN_SAME_DIRECTORY.json'
		self.bucket_name = 'tumdilabbucket86419_data'

        try:
			self.credentials = service_account.Credentials.from_service_account_file(pathToFiles+self.SERVICE_ACCOUNT_FILE)
			self.credentials = self.credentials.with_scopes(SCOPES)
		except FileNotFoundError as e:
			self.credentials = None


    def getImage(self,someParameters):
        pass
        
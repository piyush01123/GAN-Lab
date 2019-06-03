
import os

def copy_to_gcs(local_file, gcs_file):
    os.system('gsutil cp %s %s' %(local_file, gcs_file))

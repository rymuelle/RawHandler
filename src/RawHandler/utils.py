import requests


def download_file_requests(url, local_filename):
    """
    Downloads a file from a given URL using the requests library.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The desired local filename to save the downloaded file as.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"File '{local_filename}' downloaded successfully from '{url}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

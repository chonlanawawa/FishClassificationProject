from pandas import read_csv
from fastbook import download_url
from csv import reader
import logging

# read urls.csv file and make it to list (array)
with open("urls.csv") as file:
    urls = list(reader(file, delimiter=","))

# read labels file
df = read_csv("df.csv")

# make logging for easier moderation
logging.basicConfig(filename="download.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# make a funcion to download the image
def download_batch(start, end):
  n = 0
  try:
    for i in range(start, end):
      logging.info(f"Currently at index : {i}")
      for j in range(0,120):
        try:
          download_url(urls[i][j], f'images/{df.loc[i, "scientific"]}/{j}.jpg',show_progress=False)
        except Exception as e:
          logging.info(f"Error occur at url : {urls[i][j]}")
          n += 1
    logging.info(f"Download successfully at index : {i} \nFailed download count : {n}")
  except KeyboardInterrupt:
    logging.info(f"Stop by user at index : {i} \nFailed download count : {n}")

# start download
logging.info("Start Running")
download_batch(93,131)
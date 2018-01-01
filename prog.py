import unicodedata
import wget
import os
import datetime
import json
from time import gmtime, strftime

cwd = os.getcwd()

api_file = open("api.txt")
key = api_file.read()

#https://stackoverflow.com/questions/319426/how-do-i-do-a-case-insensitive-string-comparison/29247821
def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

    
def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)

    
def sanitize_query(text):
    #https://stackoverflow.com/questions/2365411/python-convert-unicode-to-ascii-without-errors
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    text = text.replace(" ", "+")
    return text

    
def search(url):
    time = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
    out_file = "search_result_" + time
    wget.download(url, out_file)
    return out_file

    
def find_title_in_json(file, title):
    myfile = open(file)
    j = json.loads(myfile.read())
    if (j['total_results'] == 1):
        result_ind = 0
    j = j['results'][result_ind]
    if caseless_equal(title, j["title"]):
        return j
    return("error")


def get_json(title):
    url="https://api.themoviedb.org/3/search/movie?api_key=" + key + "&query="
    url = url + title
    #out_file = search(url)
    out_file="search_result"
    j = find_title_in_json(out_file, title)
    print(j["poster_path"])

get_json("ratatouille")
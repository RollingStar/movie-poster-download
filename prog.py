import unicodedata
import wget
import os
import datetime
from time import gmtime, strftime

cwd = os.getcwd()

api_file = open("api.txt")
key = api_file.read()
print(key)

def sanitize_query(text):
    #https://stackoverflow.com/questions/2365411/python-convert-unicode-to-ascii-without-errors
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    text = text.replace(" ", "+")
    return text

def search(url):
    time = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
    out_file = cwd + "_" + time
    wget.download(url, out_file)
    response = open(out_file)
    print(response)
    #or result in results:
    #    if result.title == title:
    #        return result

url="https://api.themoviedb.org/3/search/movie?api_key=" + key + "&query="
search(url)

def get_json(title):
    url="https://api.themoviedb.org/3/search/movie?api_key=" + key + "&query="
    url = url + title
    json = wget(url, title)
    return json

def get_poster(title, rating):
    json = get_json(title)
    movie_num = json.movie_num
    url = "https://api.themoviedb.org/3/movie/" + movie_num + "?api_key=" + key
    #json = do_wget(url)
    #poster_url = 
    #wget url/150/json.url -> /rating/title.jpg

#title, link, rating
#if json.title == title:
#	wget url/150/json.url -> /rating/title.jpg
	
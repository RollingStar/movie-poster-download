import unicodedata
import wget
import os
import datetime
import json
import unicodecsv
import urllib
import warnings
from time import gmtime, strftime

# https://developers.themoviedb.org/3/configuration/get-api-configuration
poster_width = str(154)

cwd = os.getcwd()

api_file = open("api.txt")
key = api_file.read()
BASE_URL="https://api.themoviedb.org/3/search/movie?api_key=" + key + "&query="

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

    
def search_movie(movie_title):
    #might or may not make sense to normalize the title before we search
    url = BASE_URL + movie_title.replace(" ", "+")
    out_file = "search_result_" + str(movie_title) + ".json"
    #don't download the json if we already have results for the same search
    if os.path.isfile(out_file):
        return out_file
    wget.download(url, out_file)
    return out_file

    
def find_title_in_json(json_file, movie_title):
    myfile = open(json_file, encoding="utf-8")
    j = json.loads(myfile.read())
    
    def iterate_through_results(json_file, movie_title):
        max_ind = json_file['total_results']
        for i in range(0, max_ind):
            print(i)
            if caseless_equal(movie_title, json_file['results'][i]["title"]):
                return i
            #we couldn't find a match
            return -1
            
    if (j['total_results'] == 1):
        result_ind = 0
    else:
        print("multi results")
        result_ind = iterate_through_results(j, movie_title)
    if result_ind == -1:
        warnings.warn("title match not found in JSON search results. possible foreign language title?")
        #just return the first result and hope for the best
        result_ind = 0
    single_json = j['results'][result_ind]
    if caseless_equal(movie_title, single_json["title"]):
        return single_json
    return single_json

def get_json(movie_title):
    search_result = search_movie(movie_title)
    j = find_title_in_json(search_result, movie_title)
    return j

def download_poster(movie_title, rating=None):
    j = get_json(movie_title)
    poster_url = "https://image.tmdb.org/t/p/w" #+ poster_width #+ j["poster_path"]
    print(poster_url)
    if rating:
        out_filename = "\\" + str(rating) + "\\" + j["title"] + ".jpg"
    else:
        out_filename = j["title"] + ".jpg"
    if os.path.isfile(out_filename) == False:
        wget.download(poster_url, out_filename)

def download_posters(movies_csv):
#movies_csv is a list of movie titles alongside their rating.    
    with open(movies_csv, 'rb') as f_input:
        movies = unicodecsv.reader(f_input, encoding='utf-8-sig', delimiter=',', quotechar='"')
        for movie, rating in movies:
            if movies.line_num <= 10:
                print(movie + " " + rating)
                download_poster(movie)
            #download_poster(movie, rating)

download_posters("movies.csv")
import unicodedata
import unidecode
import math
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

#https://stackoverflow.com/questions/319426/how-do-i-do-a-case-insensitive-string-comparison/29247821
def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

    
def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)

#should be made more robust, but this fits my needs thus far
def sanitize_filename(text):
    return text.replace(":", "-")

    
def search_movie(movie_title):
    #might or may not make sense to normalize the title before we search
    #
    url = BASE_URL + unidecode.unidecode(movie_title).replace(" ", "+")
    #!! need to santize ":" out of these filenames
    out_file = "search_result_" + sanitize_filename(str(movie_title)) + ".json"
    #don't download the json if we already have results for the same search
    if os.path.isfile(out_file):
        return out_file
    wget.download(url, out_file)
    return out_file

    
def find_title_in_json(json_file, movie_title, year):
    myfile = open(json_file, encoding="utf-8")
    j = json.loads(myfile.read())
    
    def iterate_through_results(json_file, movie_title, year):
        #the json index starts at 1 
        #don't bother with second page (results above 20), don't need it for my use case
        max_ind = min(json_file['total_results'] - 1, 20)
        for i in range(0, max_ind):
            if caseless_equal(movie_title, json_file['results'][i]["title"]):
                json_year = json_file['results'][i]["release_date"][0:4]
                if year == json_year:
                    return {"ind": i, "type": "title"}
        for i in range(0, max_ind):
            json_year = json_file['results'][i]["release_date"][0:4]
            if year == json_year:
                return {"ind": i, "type": "year"}
        #we couldn't find a match
        return {"ind": -1, "type": "fail"}
    if (j['total_results'] == 0):
        warn_str = "no search results found for: " + movie_title + " " + year
        warnings.warn(warn_str)
        return None
    if (j['total_results'] == 1):
        result_ind = 0
    else:
        result_dict = iterate_through_results(j, movie_title, year)
        if result_dict['type'] == 'year':
            warn_str = "match not found for title: " + movie_title + ". Match found with " + year + ". Possible foreign language title?"
            warnings.warn(warn_str)
        result_ind = result_dict['ind']
    if result_ind == -1:
        warn_str = "match not found in JSON search results for: " + movie_title + " " + year + ". Possible foreign language title?"
        warnings.warn(warn_str)
        #just return the first result and hope for the best
        result_ind = 0
    single_json = j['results'][result_ind]
    if caseless_equal(movie_title, single_json["title"]):
        return single_json
    return single_json

def get_json(movie_title, year):
    search_result = search_movie(movie_title)
    j = find_title_in_json(search_result, movie_title, year)
    return j

def download_poster(movie_title, year, rating=None):
    j = get_json(movie_title, year)
    if j is not None:
        poster_url = "https://image.tmdb.org/t/p/w" + poster_width + j["poster_path"]
        title_for_filename = j["title"]
        title_for_filename = sanitize_filename(title_for_filename)
        if rating:
            out_filename = "\\" + str(rating) + "\\" + title_for_filename + ".jpg"
        else:
            out_filename = title_for_filename + ".jpg"
        if os.path.isfile(cwd + out_filename) == False:
            wget.download(poster_url, cwd + out_filename)

def download_posters(movies_csv):
#movies_csv looks like this:
#movie_title, user_rating, release_year
    with open(movies_csv, 'rb') as f_input:
        movies = unicodecsv.reader(f_input, encoding='utf-8-sig', delimiter=',', quotechar='"')
        for movie, rating, year in movies:
            #if movies.line_num >= 33:
            rating = str(math.ceil(int(rating)/2))
            download_poster(movie, year, rating)

cwd = os.getcwd()
api_file = open("api.txt")
key = api_file.read()
BASE_URL="https://api.themoviedb.org/3/search/movie?api_key=" + key + "&query="
for i in range(1,6):
    os.makedirs(str(i), exist_ok=True)
download_posters("movies.csv")
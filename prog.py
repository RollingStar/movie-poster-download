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
from PIL import Image, ImageFont, ImageDraw
from time import gmtime, strftime

# https://developers.themoviedb.org/3/configuration/get-api-configuration
POSTER_WIDTH = str(154)
#may not be exactly corrent in all cases
POSTER_HEIGHT = math.ceil(1.5*int(POSTER_WIDTH))

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
    url = base_url + unidecode.unidecode(movie_title).replace(" ", "+")
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
            warn_str = "using year to match movie: " + movie_title + " " + year + ". Title match not found. Possible foreign language title?"
            warnings.warn(warn_str)
        result_ind = result_dict['ind']
    if result_ind == -1:
        warn_str = "No match found for: " + movie_title + " " + year + ". Returning first search result."
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
        poster_url = "https://image.tmdb.org/t/p/w" + POSTER_WIDTH + j["poster_path"]
        title_for_filename = j["title"]
        title_for_filename = sanitize_filename(title_for_filename)
        if rating:
            out_filename = "\\" + str(rating) + "\\" + title_for_filename + ".jpg"
        else:
            out_filename = title_for_filename + ".jpg"
        if not os.path.isfile(cwd + out_filename):
            wget.download(poster_url, cwd + out_filename)

def download_posters(movies_csv):
#movies_csv looks like this:
#movie_title, user_rating, release_year
    with open(movies_csv, 'rb') as f_input:
        movies = unicodecsv.reader(f_input, encoding='utf-8-sig', delimiter=',', quotechar='"')
        for movie, rating, year in movies:
            #convert imdb-like 10-point scale to a 5-star scale
            rating = str(math.ceil(int(rating)/2))
            download_poster(movie, year, rating)

#https://stackoverflow.com/questions/250357/truncate-a-string-without-ending-in-the-middle-of-a-word
def smart_truncate(content, length=100, suffix='...'):
    if len(content) <= length:
        return content
    else:
        return content[:length].rsplit(' ', 1)[0]+suffix

def make_image(folder_of_posters, num_posters_hor=5, num_posters_vert=5):
    def calculate_padding(poster_padding_decimal, num_posters, poster_size):
        print(poster_padding_decimal, num_posters, poster_size)
        pad_size = int(poster_size * poster_padding_decimal)
        print(pad_size)
        size_with_pad = int(pad_size + poster_size)
        print(size_with_pad)
        poster_positions = range(0, int(num_posters*size_with_pad), size_with_pad)
        pixels = num_posters * size_with_pad
        #make a border the same width of the padding as well
        #don't need to add border on the left/top; this is accounted for already
        print(pixels)
        pixels = pad_size + pixels
        print(pixels)
        out_dict = {"pixels": pixels, "poster_positions": poster_positions, "pad_size": pad_size}
        return out_dict
    poster_padding_w = .1
    poster_padding_vert = .15
    width_info = calculate_padding(poster_padding_w, num_posters_hor, int(POSTER_WIDTH))
    height_info = calculate_padding(poster_padding_vert, num_posters_vert, POSTER_HEIGHT)
    #pad twice for the last row of the vertical because we draw text in the padding
    height_info["pixels"] = height_info["pixels"] + height_info["pad_size"]
    #https://docs.python.org/3/library/os.html#os.scandir
    with os.scandir(folder_of_posters) as it:
        posters_per_img = num_posters_hor * num_posters_vert
        i = 0
        w = 0
        v = 0
        #transparent background
        out_img = Image.new("RGBA", (width_info["pixels"], height_info["pixels"]), color="#ffffffcc")
        font = ImageFont.truetype("arial.ttf", 20)
        for entry in it:
            # Let's hope there are no posters available with the API that are not JPGs
            if entry.is_file() and entry.name.endswith(".jpg"):
                #https://gist.github.com/glombard/7cd166e311992a828675
                if i < posters_per_img:
                    #we fill left-right, top-bottom, so v < num_posters_vert should always be True
                    if w < num_posters_hor and v < num_posters_vert:
                        in_poster = Image.open(folder_of_posters + "\\" + entry.name)
                        w_start = width_info["poster_positions"][w] + width_info["pad_size"]
                        v_start = height_info["poster_positions"][v] + height_info["pad_size"]
                        w_end = in_poster.size[0] + width_info["poster_positions"][w] + width_info["pad_size"]
                        v_end = in_poster.size[1] + height_info["poster_positions"][v] + height_info["pad_size"]
                        out_img.paste(in_poster, (w_start, v_start, w_end, v_end))
                        draw = ImageDraw.Draw(out_img)
                        title_print = entry.name[:-4]
                        title_print = smart_truncate(title_print, 15)
                        #draw text in the same spot, regardless of poster height.
                        #could break if you have a really tall poster, but then you have other problems
                        v_end_for_text = .2 * height_info["pad_size"] + POSTER_HEIGHT + height_info["poster_positions"][v] + height_info["pad_size"]
                        draw.text((w_start, v_end_for_text), title_print, font=font, fill="black")
                        i = i+1
                        w = w+1
                        if w >= num_posters_hor:
                            v = v+1
                            w = 0
                    else:
                        #shouldn't happen
                        #new line
                        v = v+1
                else:
                    i = 0
                    w = 0
                    v = 0
                    out_img.show()
                    #need to make a function out of the image creation logic
                    #otherwise we skip posters that trigger the reset
                    out_img = Image.new("RGBA", (width_info["pixels"], height_info["pixels"]), color="#ffffffcc")
        #if we don't fill every row, crop image after we make it
        # we do i+1 even after the last image
        i = i-1
        if i <= num_posters_hor * (num_posters_vert - 1):
            #!! off by one error here, also all the cropped posters are cropped too short, probable culprit is crop_v
            #no crop on width
            crop_w = width_info["pixels"]
            v = max(math.ceil(i/num_posters_hor) - 1, 0)
            crop_v = .2 * height_info["pad_size"] + POSTER_HEIGHT + height_info["poster_positions"][v] + height_info["pad_size"]
            out_img = out_img.crop((0, 0, crop_w, crop_v))
        print(i)
        print(out_img)
        out_img.show()
        
        
cwd = os.getcwd()
api_file = open("api.txt")
key = api_file.read()
base_url="https://api.themoviedb.org/3/search/movie?api_key=" + key + "&query="
for i in range(1,6):
    os.makedirs(str(i), exist_ok=True)
    make_image(str(i))
#download_posters("movies.csv")
#make_image("5")
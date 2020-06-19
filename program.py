'''Download movie posters and make images from them.'''
import math
import os
import json
import warnings
import shutil
import unicodedata
import unidecode
import logging
import wget
# pillow, fork of PIL
from PIL import Image, ImageFont, ImageDraw
import pdb
import pandas as pd
import re
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cwd = os.getcwd()

MAIN_DIR = cwd
KEY = open(os.path.join(MAIN_DIR, "api.txt")).read()
CSV_FILE = os.path.join(MAIN_DIR, "ratings.csv")
POSTER_DIR = os.path.join(MAIN_DIR, "posters")
JSON_DIR = os.path.join(MAIN_DIR, 'json')
OUTPUT_DIR = os.path.join(MAIN_DIR, 'output')

MIN_DATE = "2020-01-01"
MAX_DATE = "2099-12-31"
FILENAME_PREFIX = datetime.today().strftime('%Y-%m-%d')

# possible types are ['movie', 'short', 'tvEpisode', 'tvMiniSeries', 'tvMovie', 'tvSeries', 'tvShort', 'tvSpecial', 'video']
WANTED_TYPES = ['movie', 'tvMovie', 'tvSpecial', 'video']
# does the header depend on the language of the user requesting the CSV?
CSV_HEADER = "Const,Your Rating,Date Rated,Title,URL,Title Type,IMDb Rating,Runtime (mins),Year,Genres,Num Votes,Release Date,Directors"
# https://developers.themoviedb.org/3/configuration/get-api-configuration
POSTER_WIDTH = str(154)
# not be exactly correct in all cases
# there's code in make_single_poster to center short posters vertically
POSTER_HEIGHT = math.ceil(1.5 * int(POSTER_WIDTH))
POSTERS_PER_ROW = 5
POSTERS_PER_COL = 5
POSTERS_PER_PAGE = POSTERS_PER_ROW * POSTERS_PER_COL
V_PADDING = .15
W_PADDING = .1

MAIN_FONT = ImageFont.truetype("arial.ttf", 20)
STAR_FONT = ImageFont.truetype("seguisym.ttf", 70)
# "#D8F6FF" is a light blue
BG_COLOR = "#D8F6FF"
TRANSPARENT = "#FFFFFF00"


def good_imdb_header(head_str):
    return (head_str == CSV_HEADER)


def five_star_scale(rating):
    # convert imdb-like 10-point scale to a 5-star scale
    return(str(math.ceil(int(rating) / 2)))


def row_dl_poster(df_row):
    new_title = download_poster(imdb_id=str(df_row['Const']),
                                movie_title=str(df_row['movie']), year=str(df_row['year']),
                                rating=str(df_row['rating']),
                                base_filename=df_row['base_filename'])
    # new_title from the JSON, often translated
    return(new_title)


# https://github.com/beetbox/beets/blob/2b8a2eb96bcaf418cdfcc0cfe762a18e31cff138/beetsplug/the.py#L66
def unthe(text, pattern):
    """Moves pattern in the path format string
    text -- text to handle
    pattern -- regexp pattern (case ignore is already on)
    """
    if text:
        r = re.compile(pattern, flags=re.IGNORECASE)
        try:
            t = r.findall(text)[0]
        except IndexError:
            return text
        else:
            r = re.sub(r, '', text).strip()
            # put the articles after the main text, with a comma ","
            fmt = '{0}, {1}'
            return fmt.format(r, t.strip()).strip()
    else:
        return u''


def title_sort(df):
    '''Sort a title by special criteria.
    (ex., put articles ["A" "An" "The"] at the end)'''
    PATTERN_THE = u'^[the]{3}\\s'
    PATTERN_A = u'^[a][n]?\\s'
    pats = [PATTERN_THE, PATTERN_A]
    sorted_title = df[['movie']][0]
    for pat in pats:
        sorted_title = unthe(sorted_title, pat)
    return(sorted_title)


def get_neg_rating(rating):
    return (-1 * int(rating))


def imdb_csv_to_pandas(file):
    with open(file, 'r') as f:
        d = f.read()
        head_str = d.split('\n', 1)[0]
        if not good_imdb_header(head_str):
            log.warning("Header:\n{}\ndoes not match expectation:\n{}".format(
                head_str, CSV_HEADER))
    df = pd.read_csv(file, encoding='latin-1')
    df = df[df['Title Type'].isin(WANTED_TYPES)]
    df = df.rename(columns={'Title': 'movie',
                            'Year': 'year', 'Date Rated': 'date_rated'})
    df = df.astype({'year': 'int64'})
    df['rating'] = df['Your Rating'].map(five_star_scale)
    # to be sorted on in pandas group by. tried something more logical, didn't work.
    df['negative_rating'] =df['rating'].map(get_neg_rating)
    df['base_filename'] = df[['movie', 'year', 'Const']].apply(
        func=filename_from_df, axis='columns')
    df['date_rated'] = pd.to_datetime(df['date_rated'])
    df = df.loc[df['date_rated'] >= pd.to_datetime(MIN_DATE)]
    df = df.loc[df['date_rated'] <= pd.to_datetime(MAX_DATE)]
    
    df['movie'] = df[['Const', 'movie', 'year', 'rating', 'base_filename']].apply(func=row_dl_poster,  axis='columns')
    df = df[['movie', 'year', 'rating', 'base_filename', 'negative_rating']]
    return(df)

# https://stackoverflow.com/questions/319426/how-do-i-do-a-case-insensitive-string-comparison/29247821


def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())


def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)


# https://github.com/django/django/blob/master/django/utils/text.py#L394
def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def filename_from_df(df):
    base_str = df['movie'] + ' ' + str(df['year']) + ' ' + df['Const']
    return(slugify(base_str))


def search_movie(imdb_id, movie_title, base_filename, json_dir=JSON_DIR):
    if imdb_id:
        url = 'https://api.themoviedb.org/3/find/' + imdb_id + \
            '?api_key=' + KEY + '&external_source=imdb_id'
    else:
        # might or may not make sense to normalize the title before we search
        base_url = "https://api.themoviedb.org/3/search/movie?api_key=" + \
            KEY + "&query="
        url = base_url + unidecode.unidecode(movie_title).replace(" ", "+")
    os.makedirs(json_dir, exist_ok=True)
    json_filename = os.path.join(json_dir, base_filename + ".json")
    # see if it's cached
    if os.path.isfile(json_filename):
        return json_filename
    wget.download(url, json_filename)
    return json_filename


def find_title_in_results(json_file, movie_title, year):
    myfile = open(json_file, encoding="utf-8")
    j = json.loads(myfile.read())

    def iterate_through_results(json_file, movie_title, year):
        # the json index starts at 1
        # don't bother with second page (results above 20)
        max_ind = min(json_file['total_results'] - 1, 20)
        log.debug(movie_title)
        for m in range(0, max_ind):
            log.debug(json_file['results'][m]["title"])
            if caseless_equal(movie_title, json_file['results'][m]["title"]):
                json_year = json_file['results'][m]["release_date"][0:4]
                if year == json_year:
                    return {"ind": m, "type": "title"}
        for m in range(0, max_ind):
            json_year = json_file['results'][m]["release_date"][0:4]
            if year == json_year:
                return {"ind": m, "type": "year"}
        # we couldn't find a match
        pdb.set_trace()
        return {"ind": -1, "type": "fail"}
    if "movie_results" in j:
        # exact IMDB id search, slightly different JSON format(?)
        if len(j["movie_results"]) != 1:
            log.warning("{} ({}): Expected 1 IMDB id search result, got {}".format(movie_title, year, len(j["movie_results"])))
        if len(j["movie_results"]) < 1:
            return None
        single_json = j["movie_results"][0]
        return(single_json)
    if j['total_results'] == 0:
        warn_str = "no search results found for: " + movie_title + " " + year
        log.warning(warn_str)
        return None
    if j['total_results'] == 1:
        result_ind = 0
    else:
        result_dict = iterate_through_results(j, movie_title, year)
        if result_dict['type'] == 'year':
            warn_str = "{}: Title match not found.".format(movie_title) + \
                "Possible foreign language title. " + \
                "Using year {} to match.".format(year)
            log.warning(warn_str)
        result_ind = result_dict['ind']
    if result_ind == -1:
        pdb.set_trace()
        warn_str = "{} ({}): No match found.".format(
            movie_title, year) + " Returning first search result."
        log.warning(warn_str)
        result_ind = 0
    single_json = j['results'][result_ind]
    if caseless_equal(movie_title, single_json["title"]):
        return single_json
    return single_json


def get_json(imdb_id, movie_title, year, base_filename):
    json_filename = search_movie(imdb_id, movie_title, base_filename)
    single_json = find_title_in_results(json_filename, movie_title, year)
    return single_json


def download_poster(movie_title, year, base_filename,
    rating=None, poster_dir=POSTER_DIR, imdb_id=None):

    def copy_none_poster():
        log.warning("{} ({}): no poster found; using none.jpg".format(movie_title, year))
        none_path = os.path.join(POSTER_DIR, 'none.jpg')
        shutil.copy(none_path, out_filename)

    os.makedirs(POSTER_DIR, exist_ok=True)
    out_filename = os.path.join(POSTER_DIR, base_filename + ".jpg")
    j = get_json(imdb_id, movie_title, year, base_filename)

    if j is not None:
        if j["poster_path"] is None:
            # maybe we manually put a poster in the right spot
            if os.path.exists(out_filename):
                logging.warning("Poster on filesystem but not online; did you manually put it there?")
                return(j["title"])
            else:
                copy_none_poster()
        else:
            poster_url = "https://image.tmdb.org/t/p/w" + \
                POSTER_WIDTH + j["poster_path"]
            if not os.path.isfile(out_filename):
                wget.download(poster_url, out_filename)
        return(j["title"])
    else:
        copy_none_poster()
        return(movie_title)


# https://stackoverflow.com/questions/250357/truncate-a-string-without-ending-in-the-middle-of-a-word
def smart_truncate(content, length=100, suffix='...'):
    if len(content) <= length:
        return content
    else:
        return content[:length].rsplit(' ', 1)[0] + suffix



def make_single_poster(df, folder_of_posters, font):
    '''combine a poster with its label text.
    `df`: a row of the DF (one movie).'''
    in_poster = Image.open(os.path.join(folder_of_posters, df['base_filename'] + '.jpg'))
    width = in_poster.width
    in_height = in_poster.height
    scale_factor = in_height/int(POSTER_HEIGHT)
    target_width = math.floor(width / scale_factor)
    # shrink posters that are too big, preserving aspect ratio
    in_poster = in_poster.resize((int(target_width), int(POSTER_HEIGHT)))
    width = in_poster.width
    in_height = in_poster.height
    height = int(POSTER_HEIGHT * (1 + V_PADDING))
    # center too-short posters
    start_y = math.floor((POSTER_HEIGHT - in_height)/2)
    out_img = Image.new("RGBA", size=(int(POSTER_WIDTH), height), color=BG_COLOR)
    # in_poster = in_poster.crop((0, start_y, int(POSTER_WIDTH), int(POSTER_HEIGHT)))
    # make a blank slate with BG_COLOR so the text is cropped at POSTER_WIDTH, not sooner if the poster is too narrow
    # temp_canvas = Image.new("RGBA", size=(int(POSTER_WIDTH), int(POSTER_HEIGHT)), color=BG_COLOR)
    # temp_canvas.paste(in_poster, (0, start_y))
    # if (smart_truncate(df['movie'], 15) == "Heavy Metal..."):
    #     temp_canvas.show()
    #     in_poster.show()
    out_img.paste(in_poster, (0, start_y))
    draw = ImageDraw.Draw(out_img)
    title_print = smart_truncate(df['movie'], 15)
    # align text to the same spot, regardless of few-pixel
    # variations in individual poster height.
    text_y = int(POSTER_HEIGHT * (1 + .03))
    draw.text((0, text_y), title_print, font=font, fill='black')
    return(out_img)


def make_row(sliced_df):
    '''return row that may or may not be combined into a large image.
    No left margin (create one when calling `make_row()`).
    `sliced_df`: a slice of the df with nrow(sliced_df) <= max_posters_per_row '''
    # 10% padding to right of poster
    # 15% padding before first row and after last row
    # 0% padding between rows (the text is sort of a padding)
    next_x = 0
    width_plus_pad = int(POSTER_WIDTH) + math.floor(W_PADDING * int(POSTER_WIDTH))
    total_width = width_plus_pad * POSTERS_PER_ROW
    total_height = math.floor(POSTER_HEIGHT * (1 + V_PADDING))
    next_y = 0
    out_img = Image.new("RGBA", size=(total_width, total_height), color=BG_COLOR)
    ii = 0
    for index, row in sliced_df.iterrows():
        next_poster = make_single_poster(row, POSTER_DIR, MAIN_FONT)
        out_img.paste(next_poster, (next_x, next_y))
        ii = ii + 1
        next_x = ii * width_plus_pad
    return out_img


def make_sub_images(df, header_text = None):
    '''return sub image(s) that may or may not be combined into a large image.
    ex., all 5 star movies are made here, and may be joined with a
    sub-image of all the 4-star movies.
    `header_txt`: A label drawn above the image.
    Blank space is added above the image if header_text exists. '''
    
    def init_page(df, header_text=header_text):
        pad_w = math.floor(W_PADDING * int(POSTER_WIDTH))
        next_x = pad_w
        next_y = 0
        width_plus_pad = int(POSTER_WIDTH) + pad_w
        total_width = width_plus_pad * POSTERS_PER_ROW
        # pad on the left of the first poster. right pad is already accounted for
        total_width = total_width + pad_w
        height_plus_pad = int(POSTER_HEIGHT * (1 + V_PADDING))
        # crop blank parts of image if there aren't many movies
        # ex., <6 movies
        if len(df) < POSTERS_PER_PAGE:
            total_height = height_plus_pad * math.ceil(len(df) / POSTERS_PER_COL)
            total_width = width_plus_pad * min(len(df), POSTERS_PER_ROW) + pad_w
            # don't make something smaller than the header_text
            if header_text:
                # 2 * width_plus_pad is about 1.5x wider than the 5 star label I use
                total_width = max(total_width, width_plus_pad * 2 + pad_w)
        else:
            total_height = height_plus_pad * POSTERS_PER_COL
        if header_text:
            # leave extra space for a bigger header than the caption text
            # for each poster
            # magic number, works fine with the default settings
            pad_height = math.floor(POSTER_HEIGHT * .4)
            total_height = total_height + pad_height
            next_y = pad_height
        out_img = Image.new("RGBA", (total_width, total_height), color=BG_COLOR)
        if header_text:
            draw = ImageDraw.Draw(out_img)
            # star text is weird, what looks nice and centered is not
            # what is mathematically centered
            # looks good with "seguisym.ttf" at 70 pt font.
            # anything else probably needs to be changed.
            # negative 0.07 starts the text outside the image(?),
            # but the star text is very low (???) so it works out.
            pad_y = math.floor(STAR_FONT.size * -.07)
            draw.text((pad_w, pad_y), header_text, font=STAR_FONT, fill='black')
        return(out_img, next_x, next_y, height_plus_pad)

    for ii, pagedf in df.groupby(np.arange(len(df)) // POSTERS_PER_PAGE):
        out_page, next_x, next_y, height_plus_pad = init_page(df=pagedf)
        for jj, rowdf in pagedf.groupby(np.arange(len(pagedf)) // POSTERS_PER_ROW):
            row = make_row(rowdf)
            out_page.paste(row, (next_x, next_y))
            next_y = height_plus_pad + next_y
        # will break on >25 (>POSTERS_PER_PAGE) posters
        return(out_page)
        

def add_watermark(main_img, text = 'github.com/RollingStar/movie-poster-download'):
    '''Add text watermark to image at the bottom-right.'''
    # make a bigger canvas for the watermark
    bigger_canvas = Image.new("RGBA", (main_img.width, (20 + main_img.height)), color=BG_COLOR)
    bigger_canvas.paste(main_img, (0, 0))
    draw = ImageDraw.Draw(bigger_canvas)
    myfont = ImageFont.truetype("arial.ttf", 12)
    # found through trial and error of Arial at 12 point size
    x_pixels_needed = 265
    draw.multiline_text(((main_img.width - x_pixels_needed), main_img.height), text,
        font=myfont, fill='black', align="left")
    return(bigger_canvas)

def make_images_by_rating(df):
    '''Split `df` into rating categories and draw pages with rating
    text on each one.'''
    filled_star = '★'
    empty_star = '☆'
    sub_imgs = []
    for ii, subdf in df.groupby(by='negative_rating'):
        rating = int(subdf.iloc[0]['rating'])
        rating_text = filled_star * rating
        rating_text = rating_text + (empty_star * (5 - rating))
        sorted_df = subdf.sort_values(by='title_sort')
        sub_img = make_sub_images(subdf, header_text=rating_text)
        sub_imgs.append(sub_img)
    # combine multiple sub-images
    max_y = POSTER_HEIGHT * (POSTERS_PER_COL + 1)
    total_width = 0
    total_height = 0
    for img in sub_imgs:
        total_height = total_height + img.height
        total_width = max(total_width, img.width)
    main_img = Image.new("RGBA", (total_width, total_height), color=BG_COLOR)
    height_so_far = 0
    for img in sub_imgs:
        main_img.paste(img, (0, height_so_far))
        # start the next paste below the current one
        height_so_far = img.height + height_so_far
    final_img = add_watermark(main_img)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'all-movies-' + FILENAME_PREFIX + '.png')
    final_img.save(out_path)
    return final_img

  
def postprocess_df(df):
    df['title_sort'] = df[['movie']].apply(func=title_sort, axis='columns')
    # sort the df how we want it
    df = df.sort_values(by="title_sort")
    # save a LOT of trouble. remove indexes to dropped rows
    # (ex., rows that are outside the date range)
    df.reset_index(drop = True, inplace=True)
    return(df)


df = imdb_csv_to_pandas(CSV_FILE)
# do stuff we can only do after we get the JSONs
df = postprocess_df(df)
make_images_by_rating(df)
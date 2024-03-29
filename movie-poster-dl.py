"""Download movie posters and make images from them."""
import math
import os
import json
import shutil
import argparse
import unicodedata
import unidecode
import logging
import wget
import os
from pathlib import Path

# pillow, fork of PIL
from PIL import Image, ImageFont, ImageDraw
import pdb
import pandas as pd
import re
import numpy as np
from datetime import datetime


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# init dates
now = datetime.today()

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
KEY = open(os.path.join(MAIN_DIR, "api.txt")).read()
# CSV_FILE = os.path.join(MAIN_DIR, "ratings.csv")
POSTER_DIR = os.path.join(MAIN_DIR, "posters")
JSON_DIR = os.path.join(MAIN_DIR, "json")
OUTPUT_DIR = os.path.join(MAIN_DIR, "output")
FILENAME_PREFIX = now.strftime("%Y%m%d-%H%M%S")
EXPORT_FILENAME = os.path.join(MAIN_DIR, "export" + "_" + FILENAME_PREFIX + ".csv")
# possible types are ['movie', 'short', 'tvEpisode', 'tvMiniSeries', 'tvMovie', 'tvSeries', 'tvShort', 'tvSpecial', 'video']
WANTED_TYPES = ["movie", "tvMovie", "tvSpecial", "video"]
# does the header depend on the language of the user requesting the CSV?
CSV_HEADER = "Const,Your Rating,Date Rated,Title,URL,Title Type,IMDb Rating,Runtime (mins),Year,Genres,Num Votes,Release Date,Directors"
# https://developers.themoviedb.org/3/configuration/get-api-configuration
POSTER_WIDTH = 154
# not be exactly correct in all cases
# there's code in make_single_poster to center short posters vertically
POSTER_HEIGHT = math.ceil(1.5 * POSTER_WIDTH)
POSTERS_PER_ROW = 5
POSTERS_PER_COL = 5
POSTERS_PER_PAGE = POSTERS_PER_ROW * POSTERS_PER_COL
V_PADDING = 0.15
W_PADDING = 0.1

MAIN_FONT = ImageFont.truetype("arial.ttf", 20)
STAR_FONT = ImageFont.truetype("seguisym.ttf", 70)
# "#D8F6FF" is a light blue
BG_COLOR = "#D8F6FF"
TRANSPARENT = "#FFFFFF00"


def good_imdb_header(head_str):
    return head_str == CSV_HEADER


def five_star_scale(rating, scale=5):
    imdb_scale = 10
    scale_factor = imdb_scale / scale
    # convert imdb-like 10-point scale to an X-star scale
    return str(math.ceil(int(rating) / scale_factor))


def row_dl_poster(df_row):
    new_title = download_poster(
        imdb_id=str(df_row["Const"]),
        movie_title=str(df_row["movie"]),
        year=str(df_row["year"]),
        rating=str(df_row["rating"]),
        base_filename=df_row["base_filename"],
    )
    # new_title from the JSON, often translated
    return new_title


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
            r = re.sub(r, "", text).strip()
            # put the articles after the main text, with a comma ","
            fmt = "{0}, {1}"
            return fmt.format(r, t.strip()).strip()
    else:
        return u""


def title_sort(df):
    """Sort a title by special criteria.
    (ex., put articles ["A" "An" "The"] at the end)"""
    PATTERN_THE = u"^[the]{3}\\s"
    PATTERN_A = u"^[a][n]?\\s"
    pats = [PATTERN_THE, PATTERN_A]
    sorted_title = df[["movie"]][0]
    for pat in pats:
        sorted_title = unthe(sorted_title, pat)
    return sorted_title


def get_neg_rating(rating):
    return -1 * int(rating)


def imdb_csv_to_pandas(file, scale=5, genre=None):
    with open(file, "r") as f:
        d = f.read()
        head_str = d.split("\n", 1)[0]
        if not good_imdb_header(head_str):
            log.warning(
                "Header:\n{}\ndoes not match expectation:\n{}".format(
                    head_str, CSV_HEADER
                )
            )
    df = pd.read_csv(file, encoding="latin-1")
    log.debug(len(df))
    df = df[df["Title Type"].isin(WANTED_TYPES)]
    log.debug(len(df))
    if genre:
        df = df[df["Genres"].str.contains(genre)]
        log.debug(len(df))
    df = df.rename(
        columns={
            "Title": "movie",
            "Year": "year",
            "Date Rated": "date_rated",
            "Runtime (mins)": "runtime",
        },
        inplace=False,
    )
    log.debug(len(df))
    df = df.astype({"year": "int64"})
    df["rating"] = df["Your Rating"].apply(five_star_scale, args=[scale])
    # to be sorted on in pandas group by. tried something more logical, didn't work.
    df["negative_rating"] = df["rating"].map(get_neg_rating)
    df["base_filename"] = df[["movie", "year", "Const"]].apply(
        func=filename_from_df, axis="columns"
    )
    df["date_rated"] = pd.to_datetime(df["date_rated"])
    # need to subset after looking backwards for rewatches
    # df = df.loc[df['date_rated'] >= pd.to_datetime(MIN_DATE)]
    # df = df.loc[df['date_rated'] <= pd.to_datetime(MAX_DATE)]
    df = df.loc[df["runtime"] >= 40]
    # TV compilation incorrectly tagged as a video
    df = df.loc[df["movie"] != "Strongbad_email.exe"]
    df["movie"] = df[["Const", "movie", "year", "rating", "base_filename"]].apply(
        func=row_dl_poster, axis="columns"
    )
    df = df[
        [
            "movie",
            "year",
            "rating",
            "base_filename",
            "negative_rating",
            "date_rated",
            "Const",
        ]
    ]
    if len(df) <= 0:
        Exception("DF has no rows.")
    return df


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
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def filename_from_df(df):
    base_str = df["movie"] + " " + str(df["year"]) + " " + df["Const"]
    return slugify(base_str)


def search_movie(imdb_id, movie_title, base_filename, json_dir=JSON_DIR):
    if imdb_id:
        url = (
            "https://api.themoviedb.org/3/find/"
            + imdb_id
            + "?api_key="
            + KEY
            + "&external_source=imdb_id"
        )
    else:
        # might or may not make sense to normalize the title before we search
        base_url = (
            "https://api.themoviedb.org/3/search/movie?api_key=" + KEY + "&query="
        )
        url = base_url + unidecode.unidecode(movie_title).replace(" ", "+")
    os.makedirs(json_dir, exist_ok=True)
    json_filename = os.path.join(json_dir, base_filename + ".json")
    cached = os.path.isfile(json_filename)
    if cached:
        return json_filename
    else:
        wget.download(url, json_filename)
        log.debug("\n")  # wget logs to debug without /n
        return json_filename


def find_title_in_results(json_file, movie_title, year):
    myfile = open(json_file, encoding="utf-8")
    j = json.loads(myfile.read())

    def iterate_through_results(json_file, movie_title, year):
        # the json index starts at 1
        # don't bother with second page (results above 20)
        max_ind = min(json_file["total_results"] - 1, 20)
        log.debug(movie_title)
        for m in range(0, max_ind):
            log.debug(json_file["results"][m]["title"])
            if caseless_equal(movie_title, json_file["results"][m]["title"]):
                json_year = json_file["results"][m]["release_date"][0:4]
                if year == json_year:
                    return {"ind": m, "type": "title"}
        for m in range(0, max_ind):
            json_year = json_file["results"][m]["release_date"][0:4]
            if year == json_year:
                return {"ind": m, "type": "year"}
        log.warning("we couldn't find a match")
        pdb.set_trace()
        return {"ind": -1, "type": "fail"}

    if "movie_results" in j:
        # exact IMDB id search, slightly different JSON format(?)
        if len(j["movie_results"]) != 1:
            log.warning(
                "{} ({}): Expected 1 IMDB id search result, got {}".format(
                    movie_title, year, len(j["movie_results"])
                )
            )
        if len(j["movie_results"]) < 1:
            return None
        single_json = j["movie_results"][0]
        return single_json
    if j["total_results"] == 0:
        warn_str = "no search results found for: " + movie_title + " " + year
        log.warning(warn_str)
        return None
    if j["total_results"] == 1:
        result_ind = 0
    else:
        result_dict = iterate_through_results(j, movie_title, year)
        if result_dict["type"] == "year":
            warn_str = (
                "{}: Title match not found.".format(movie_title)
                + "Possible foreign language title. "
                + "Using year {} to match.".format(year)
            )
            log.warning(warn_str)
        result_ind = result_dict["ind"]
    if result_ind == -1:
        pdb.set_trace()
        warn_str = (
            "{} ({}): No match found.".format(movie_title, year)
            + " Returning first search result."
        )
        log.warning(warn_str)
        result_ind = 0
    single_json = j["results"][result_ind]
    if caseless_equal(movie_title, single_json["title"]):
        return single_json
    return single_json


def get_json(imdb_id, movie_title, year, base_filename):
    json_filename = search_movie(imdb_id, movie_title, base_filename)
    single_json = find_title_in_results(json_filename, movie_title, year)
    return single_json


def download_poster(
    movie_title, year, base_filename, rating=None, poster_dir=POSTER_DIR, imdb_id=None
):
    def copy_none_poster():
        log.warning(
            "{} ({}): no poster found; using none.jpg".format(movie_title, year)
        )
        none_path = os.path.join(POSTER_DIR, "none.jpg")
        shutil.copy(none_path, out_filename)

    os.makedirs(POSTER_DIR, exist_ok=True)
    out_filename = os.path.join(POSTER_DIR, base_filename + ".jpg")
    j = get_json(imdb_id, movie_title, year, base_filename)

    if j is not None:
        if j["poster_path"] is None:
            # maybe we manually put a poster in the right spot
            if os.path.exists(out_filename):
                logging.debug(
                    "Using local filesystem poster, none found online: {}".format(
                        movie_title
                    )
                )
                return j["title"]
            else:
                copy_none_poster()
        else:
            poster_url = (
                "https://image.tmdb.org/t/p/w" + str(POSTER_WIDTH) + j["poster_path"]
            )
            if not os.path.isfile(out_filename):
                wget.download(poster_url, out_filename)
                log.debug("\n")  # wget logs to debug without /n
        return j["title"]
    else:
        copy_none_poster()
        return movie_title


def word_truncate(content):
    """Truncate on a whole word."""
    return content.rsplit(" ", 1)[0]


def make_single_poster(df, folder_of_posters, font):
    """combine a poster with its label text.
    `df`: a row of the DF (one movie)."""
    poster_path = os.path.join(folder_of_posters, df["base_filename"] + ".jpg")
    in_poster = Image.open(poster_path)
    width = in_poster.width
    in_height = in_poster.height
    scale_factor = in_height / int(POSTER_HEIGHT)
    target_width = math.floor(width / scale_factor)
    # shrink posters that are too big, preserving aspect ratio
    in_poster = in_poster.resize((int(target_width), int(POSTER_HEIGHT)))
    width = in_poster.width
    in_height = in_poster.height
    height = int(POSTER_HEIGHT * (1 + V_PADDING))
    # center too-short posters
    start_y = math.floor((POSTER_HEIGHT - in_height) / 2)
    out_img = Image.new("RGBA", size=(POSTER_WIDTH, height), color=BG_COLOR)
    # in_poster = in_poster.crop((0, start_y, POSTER_WIDTH, int(POSTER_HEIGHT)))
    # make a blank slate with BG_COLOR so the text is cropped at POSTER_WIDTH, not sooner if the poster is too narrow
    # temp_canvas = Image.new("RGBA", size=(POSTER_WIDTH, int(POSTER_HEIGHT)), color=BG_COLOR)
    # temp_canvas.paste(in_poster, (0, start_y))
    out_img.paste(in_poster, (0, start_y))
    draw = ImageDraw.Draw(out_img)
    title_print = df["movie"]
    if df["rewatch"]:
        title_print = title_print + "*"
    # deprecated
    # textw, texth = draw.textsize(title_print, font=font)
    textw = draw.textlength(title_print, font=font)
    suffix = "..."
    ii = 0
    tried_colon = False
    while textw > POSTER_WIDTH:
        if ii > 20:
            pdb.set_trace()
        rewatch_space = 0
        if df["rewatch"]:
            title_print = "*" + title_print
            rewatch_space = len("*")
        long_first_word = title_print == word_truncate(title_print)
        title_print = word_truncate(title_print)
        if long_first_word:
            chop_amt = len(suffix) + 1 + rewatch_space
            log.debug(title_print)
            title_print = title_print[:(-chop_amt)]
            if title_print.endswith(" "):
                title_print = title_print[:-1]
        if title_print.endswith(":") and (not tried_colon):
            title_print = title_print + " "
            # avoid infinite loop
            tried_colon = True
        title_print = title_print + suffix
        if df["rewatch"]:
            title_print = title_print[1:] + "*"
        # textw, texth = draw.textsize(title_print, font=font)
        textw = draw.textlength(title_print, font=font)
        ii += 1

    # textw, texth = draw.textsize(title_print, font=font)
    textw = draw.textlength(title_print, font=font)
    # align text to the same spot, regardless of few-pixel
    # variations in individual poster height.
    text_y = int(POSTER_HEIGHT * (1 + 0.03))
    draw.text((0, text_y), title_print, font=font, fill="black")
    log.debug("{} {}".format(title_print, poster_path))
    return out_img


def make_row(sliced_df):
    """return row that may or may not be combined into a large image.
    No left margin (create one when calling `make_row()`).
    `sliced_df`: a slice of the df with nrow(sliced_df) <= max_posters_per_row """
    # 10% padding to right of poster
    # 15% padding before first row and after last row
    # 0% padding between rows (the text is sort of a padding)
    next_x = 0
    width_plus_pad = POSTER_WIDTH + math.floor(W_PADDING * POSTER_WIDTH)
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


def make_sub_images(df, header_text=None):
    """return sub image(s) that may or may not be combined into a large image.
    ex., all 5 star movies are made here, and may be joined with a
    sub-image of all the 4-star movies.
    `header_txt`: A label drawn above the image.
    Blank space is added above the image if header_text exists. """
    log.debug("df len is {}".format(len(df)))

    def init_page(df, header_text=header_text):
        log.debug("df len is {}".format(len(df)))
        pad_w = math.floor(W_PADDING * POSTER_WIDTH)
        next_x = pad_w
        next_y = 0
        width_plus_pad = POSTER_WIDTH + pad_w
        total_width = width_plus_pad * POSTERS_PER_ROW
        # pad on the left of the first poster. right pad is already accounted for
        total_width = total_width + pad_w
        height_plus_pad = int(POSTER_HEIGHT * (1 + V_PADDING))
        # crop blank parts of image if there aren't many movies
        # ex., <6 movies
        if len(df) < POSTERS_PER_PAGE:
            # total_height = height_plus_pad * math.ceil(len(df) / POSTERS_PER_COL)
            total_height = height_plus_pad * math.ceil(len(df) / POSTERS_PER_ROW)
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
            pad_height = math.floor(POSTER_HEIGHT * 0.4)
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
            pad_y = math.floor(STAR_FONT.size * -0.07)
            draw.text((pad_w, pad_y), header_text, font=STAR_FONT, fill="black")
        return (out_img, next_x, next_y, height_plus_pad)

    out_pages = []
    if len(df) > POSTERS_PER_PAGE:
        extra_text = ""
        # extra_text = ' ({} Movies)'.format(len(df))
    else:
        extra_text = ""
    header_text = header_text + extra_text
    for ii, pagedf in df.groupby(np.arange(len(df)) // POSTERS_PER_PAGE):
        out_page, next_x, next_y, height_plus_pad = init_page(
            df=pagedf, header_text=header_text
        )
        for jj, rowdf in pagedf.groupby(np.arange(len(pagedf)) // POSTERS_PER_ROW):
            row = make_row(rowdf)
            out_page.paste(row, (next_x, next_y))
            next_y = height_plus_pad + next_y
        # will break on >25 (>POSTERS_PER_PAGE) posters
        out_pages.append(out_page)
        # don't redraw header_text on multiple "pages" with the same text
        header_text = None
    return out_pages


def add_watermark(
    main_img, df_len=0, text="*Rewatch. github.com/RollingStar/movie-poster-download"
):
    """Add text watermark to image at the bottom-right."""
    text = str(df_len) + " Movies." + " " + text
    # make a bigger canvas for the watermark
    bigger_canvas = Image.new(
        "RGBA", (main_img.width, (20 + main_img.height)), color=BG_COLOR
    )
    bigger_canvas.paste(main_img, (0, 0))
    draw = ImageDraw.Draw(bigger_canvas)
    myfont = ImageFont.truetype("arial.ttf", 12)
    # deprecated
    # textw, texth = draw.textsize(text, font=myfont)
    textw = draw.textlength(text, font=myfont)
    # log.debug("textw: {}".format(textw))
    # log.debug("newl: {}".format(newl))
    # deprecated
    # x_pad = draw.textsize("a", font=myfont)[0]
    pad_text = "a"  # space out the text as far as "a" is wide
    x_pad = draw.textlength(pad_text, font=myfont)
    x_pixels_needed = math.ceil(textw + (x_pad * 0.75))
    draw.multiline_text(
        ((main_img.width - x_pixels_needed), main_img.height),
        text,
        font=myfont,
        fill="black",
        align="left",
    )
    return bigger_canvas


def join_imgs(sub_imgs, write_sub_files):
    height_so_far = 0
    total_width = 0
    total_height = 0
    for img in sub_imgs:
        total_height = total_height + img.height
        total_width = max(total_width, img.width)
    main_img = Image.new("RGBA", (total_width, total_height), color=BG_COLOR)
    for jj, img in enumerate(sub_imgs):
        main_img.paste(img, (0, height_so_far))
        # start the next paste below the current one
        height_so_far = img.height + height_so_far
        if write_sub_files:
            out_path = os.path.join(
                OUTPUT_DIR, "all-movies-" + FILENAME_PREFIX + "_" + str(jj) + ".png"
            )
            img.save(out_path)
    return main_img


def make_images_by_rating(df, scale=5, write_sub_files=False):
    """Split `df` into rating categories and draw pages with rating
    text on each one."""
    filled_star = "★"
    empty_star = "☆"
    sub_imgs = []
    for ii, subdf in df.groupby(by="negative_rating"):
        rating = int(subdf.iloc[0]["rating"])
        rating_text = filled_star * rating
        rating_text = rating_text + (empty_star * (scale - rating))
        subdf.sort_values(by="title_sort")
        new_imgs = make_sub_images(subdf, header_text=rating_text)
        # temp_img = Image.new("RGBA", (total_width, total_height), color=BG_COLOR)
        # don't write sub files for ex. 30 movies in one rating set (see next call for join_imgs)
        bigger_imgs = join_imgs(new_imgs, write_sub_files=False)
        sub_imgs.append(bigger_imgs)
    # combine multiple sub-images
    # max_y = POSTER_HEIGHT * (POSTERS_PER_COL + 1)
    # total_width = 0
    # total_height = 0
    # for img in sub_imgs:
    #     total_height = total_height + img.height
    #     total_width = max(total_width, img.width)
    # now we can save sub imgs and have all films of one rating in one sub img (see above join_imgs)
    main_img = join_imgs(sub_imgs, write_sub_files)
    df_len = len(df)
    final_img = add_watermark(main_img, df_len)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "all-movies-" + FILENAME_PREFIX + ".png")
    log.debug(out_path)
    final_img.save(out_path)
    return final_img


def is_rewatch(df):
    rewatch = False
    # think there's a bug here if you only have 1 ratings.csv (no date_rated_y)
    if not (pd.isnull(df["date_rated_y"])):
        if df.date_rated != df.date_rated_y:
            rewatch = True
    return rewatch


def postprocess_df(df):
    if len(df) < 1:
        log.warning("Empty DF?")
        pdb.set_trace()
    df.drop("date_rated", axis=1)
    df["title_sort"] = df[["movie"]].apply(func=title_sort, axis="columns")
    # sort the df how we want it
    df = df.sort_values(by="title_sort")
    # save a LOT of trouble. remove indexes to dropped rows
    # (ex., rows that are outside the date range)
    df.reset_index(drop=True, inplace=True)
    rewatch_len = len(df[df["rewatch"]])
    log.info("{} movies ({} rewatches).".format(len(df.index), rewatch_len))
    return df


def get_max_date(df):
    # think there's a bug here if you only have 1 ratings.csv (no date_rated_y)
    max_date = max(df.date_rated, df.date_rated_y)
    return max_date


def save_csv(df, fname=EXPORT_FILENAME):
    tdf = df
    tdf = tdf.rename(
        columns={
            "movie": "Title",
            "year": "Year",
            "Const": "imdbID",
            "rating": "Rating10",
            "date_rated": "WatchedDate",
            "rewatch": "Rewatch",
        }
    )
    tdf = tdf[["Title", "Year", "imdbID", "Rating10", "WatchedDate", "Rewatch"]]
    tdf.to_csv(fname)
    log.info("{} saved".format(fname))


if __name__ == "__main__":
    # i think there's a bug in is_rewatch
    parser = argparse.ArgumentParser(
        description="Download movie posters and make images from them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-sd",
        default="1990-01-01",
        help="Start date (YYYY-MM-DD). Only include ratings from this date onward.",
    )
    parser.add_argument(
        "-ed",
        default="2099-12-31",
        help="End date (YYYY-MM-DD). Only include ratings from before this date.",
    )
    parser.add_argument(
        "-p",
        "--points",
        default=5,
        help="Number of points in the scale. 5=5 star, 10=10 star (IMDB/Letterboxd import) etc.",
    )
    parser.add_argument("-g", "--genre", default=None, help="genre")
    parser.add_argument(
        "-s",
        "--subfiles",
        default=False,
        help="Save a separate file for each star rating",
    )
    parser.add_argument(
        "-e",
        "--export",
        default=False,
        help="Save a CSV filtered by the app (ex., only one genre, only 40m+ films)",
    )
    parser.add_argument(
        "--skip",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Skip the final save (prep and export only)",
    )

    args = parser.parse_args()
    MIN_DATE = args.sd
    MAX_DATE = args.ed
    USER_SCALE = int(args.points)
    SAVE_SUB_FILES = args.subfiles
    # DO NOT CAST THIS TO A STRING! 'None' is not None!
    U_GENRE = args.genre
    # os.path.join(MAIN_DIR, FILENAME_PREFIX)
    paths = sorted(Path(MAIN_DIR).iterdir(), key=os.path.getmtime)
    csv_paths = []
    for path in paths:
        if "ratings" in str(path):
            if "csv" in str(path):
                csv_paths.append(path)
    df_list = []
    log.debug(csv_paths)
    for path in csv_paths:
        df_list.append(imdb_csv_to_pandas(path, scale=USER_SCALE, genre=U_GENRE))
    len_dfs = len(df_list)
    # looks like it only considers the oldest and newest file.
    # so if you want to show a movie rewatched in the same period, it will only show one of the two ratings.
    if len_dfs > 1:
        log.debug("Merging {} lists".format(len_dfs))
        combined = pd.merge(
            df_list[-1],  # last?
            df_list[0],  # first?
            how="left",
            on="Const",
            suffixes=(None, "_y"),
            validate="one_to_one",
        )
        df = combined
        log.debug("Doing rewatch calc")
        df["rewatch"] = df.apply(is_rewatch, axis=1)
    else:
        log.critical("Nothing to merge")
        # don't merge since there's nothing to merge
        df = df_list[0]
    df = df.loc[df["date_rated"] >= pd.to_datetime(MIN_DATE)]
    df = df.loc[df["date_rated"] <= pd.to_datetime(MAX_DATE)]
    # do stuff we can only do after we get the JSONs
    df = postprocess_df(df)
    df["max_date"] = df.apply(get_max_date, axis=1)
    print(args.skip)
    if not args.skip:
        make_images_by_rating(df, USER_SCALE, write_sub_files=SAVE_SUB_FILES)
    if args.export:
        save_csv(df)

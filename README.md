# movie-poster-download
Downloads posters for a list of movies, and makes group images of the posters.

The program assumes an IMDB `ratings.csv` export, but it shouldn't be too hard to rework it to other CSVs / lists.

Get a v3 API key from [The Movie DB](https://www.themoviedb.org/documentation/api) and store it as `api.txt`.

Log in to IMDB and go [here](https://www.imdb.com/list/ratings). Click the "..." in the top right to export your CSV. Sadly, user ratings can only be exported when logged-in as the user in question. The program cannot export them for you.

# Required packages

```
pip install unidecode
pip install Pillow
pip install wget
pip install requests
pip install numpy
pip install pandas
```

# Usage

```
python program.py -h
usage: program.py [-h] [-sd SD] [-ed ED]

Download movie posters and make images from them.

optional arguments:
  -h, --help  show this help message and exit
  -sd SD      Start date (YYYY-MM-DD). Only include ratings from this date onward. (default: 1990-01-01)
  -ed ED      End date (YYYY-MM-DD). Only include ratings from before this date. (default: 2099-12-31)
```

# Directory structure

Defined at the start of the script. Caches movie posters and JSON results in the current working directory (`os.getcwd()`):

* Main folder (`os.getcwd()`)
  * `posters`
  * `json` (search results)
  * `output` (final output)

# Example output

https://i.imgur.com/j6sNKMx.png

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

# Directory structure

Defined at the start of the script. Caches movie posters and JSON results in the current working directory (`os.getcwd()`):

* Main folder (`os.getcwd()`)
  * `posters`
  * `json` (search results)
  * `output` (final output)

# Example output

https://i.imgur.com/j6sNKMx.png

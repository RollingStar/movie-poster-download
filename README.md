# movie-poster-download
Downloads posters for a list of movies, and makes group images of the posters.

Get a v3 API key from [The Movie DB](https://www.themoviedb.org/documentation/api) and store it as `api.txt`.

# Required packages

```
pip install unidecode
pip install unicodecsv
pip install Pillow
pip install wget
pip install requests
  
```

# Directory structure

Defined at the start of the script. Caches movie posters and JSON results:

* Main dir (`os.getwd()`)
  * `posters`
  * `json` (search results)
  * `output` (final output)

# Example output

https://imgur.com/a/wdUPu

# weirdest-place-in-finland

Runs MaxViT-T to produce embeddings of images, scores the images according to how weird they are compared to the rest, then outputs the top 300 weirdest and non-weirdest images.

TODO rest of the readme

## Results

Download `results-jpeg.zip` from https://github.com/cxcorp/weirdest-place-in-finland/releases/tag/v0.0.1 and look at the files.

The folders have no proper naming convention, but it's something like this:

```
<maxvit|resnet>-<scoring mechanism>-<scoring mechanism details>
```

If prefixed by `ALL-`, it contains results from the FULL dataset. If not, it's from some sample of 1000-4000 images.

## Running

1. Run https://github.com/cxcorp/mml-orto-downloader-py to download image dataset
2. ctrl+f `\images\` from the repoes to change the file path where they're downloaded, and where this repo expects to find them, sorry
3. figure out what dependencies the repo needs
   - `Dockerfile` has at least some of the most essentials but there are lots more, sorry!!!
   - sorry I was too lazy with my dependency management, there is only a broken requirements.txt with lots of extraneous deps and deps from wrong indexes (e.g. pytorch needs its own index for pip)
4. make a virtualenv or something to install the deps in, install deps
5. run `python src/find_white_tiles.py` to produce `cutouts.txt` which contains list of images we will ignore because they're mostly white (regions outside borders are just white pixels)
6. run `python src/index.py` to generate embeddings out of the images into a parquet file under `parquets/`
   - this will take an hour on an RTX 3080 Ti - consider downloading a smaller sample of images first
7. run `python src/results.py` to load the parquet file and find the weirdest places according to a hybrid of two metrics:
   1. mahalanobis distance to mean embedding
   2. trains an autoencoder on all of the embeddings, then scores the images according to reconstruction error
      - training takes a few mins
8. output appears in the folder you ctrl+f'd and replaced earlier


# embedding_db

simple wrapper for extracting [inception resnet features](https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/3) from images

caches embeddings (as saved numpy arrays) in a local directory

e.g.

```
import embedding_cache
cache = embedding_cache.EmbeddingCache()
print(cache.embedding_for_file('test/eg_img.jpg').shape)  # slow first time ever run
print(cache.embedding_for_file('test/eg_img.jpg').shape)  # uses cached version
```

outputs ...

```
(1536,)
(1536,)
```

stored locally

```
$ ls .embedding_cache
eg_img.jpg.npy
```
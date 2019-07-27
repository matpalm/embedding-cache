import tensorflow as tf
import tensorflow_hub as hub
import io
import numpy as np
from PIL import Image
import os


class EmbeddingFn(object):
    def __init__(self):
        self.image_str = tf.placeholder(tf.string)
        image = tf.image.decode_jpeg(self.image_str)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, 0)  # single element batch
        resnet_features = hub.Module(
            "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/3")
        self.feature_vector = resnet_features(image)
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(),
                       tf.tables_initializer()])

    def embed(self, pil_img):
        # inception_resnet_v2 expects (299, 299)
        resized_img = pil_img.resize((299, 299))
        img_bytes = io.BytesIO()
        resized_img.save(img_bytes, format='JPEG', quality=100)
        img_bytes = img_bytes.getvalue()
        return self.sess.run(self.feature_vector,
                             feed_dict={self.image_str: img_bytes})[0]


class EmbeddingCache(object):

    def __init__(self, cache_dir='.embedding_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.embed_fn = None

    def embedding_for_file(self, img_full_fname):

        # cache based on _just_ base filename. this means we can
        # move files around between directories and keep the cached
        # version.
        just_fname = os.path.basename(img_full_fname)
        cache_file = "%s/%s.npy" % (self.cache_dir, just_fname)

        # already have it? if so just load & return
        if os.path.isfile(cache_file):
            return np.load(cache_file)

        # lazily create embedding_fn now. we do this since it's slow
        # and a lot of use cases might just using the cached dir versions
        if self.embed_fn is None:
            self.embed_fn = EmbeddingFn()

        # calculate embedding for non cacheed img
        pil_img = Image.open(img_full_fname)
        embedding = self.embed_fn.embed(pil_img)

        # cache for next time and return
        np.save(cache_file, embedding)
        return embedding

    def embedding_dim(self):
        return 1536


if __name__ == '__main__':
    cache = EmbeddingCache()
    print(cache.embedding_for_file('test/eg_img.jpg').shape)
    print(cache.embedding_for_file('test/eg_img.jpg').shape)

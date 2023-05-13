import tensorflow as tf
from tensorflow import keras
import re
import string
import einops
import tqdm

IMAGE_SHAPE = (224, 224, 3)
AUTOTUNE = tf.data.AUTOTUNE

def flickr30k(annotations, image='image', caption='caption', item=5):
    total = len(annotations) // item
    train = int(0.9 * total)
    val = int(0.95 * total)
    image, caption = annotations[image], annotations[caption]
    # train_dict = {image[item*i][:-2]: [caption[j] for j in range(item*i, item*i+item)] for i in range(train)}
    # val_dict = {image[item*i][:-2]: [caption[j] for j in range(item*i, item*i+item)] for i in range(train, val)}
    # test_dict = {image[item*i][:-2]: [caption[j] for j in range(item*i, item*i+item)] for i in range(val, total)}
    train_ds = [(image[item*i][:-2], [caption[j] for j in range(item*i, item*i+item)]) for i in range(train)]
    val_ds = [(image[item*i][:-2], [caption[j] for j in range(item*i, item*i+item)]) for i in range(train, val)]
    test_ds = [(image[item*i][:-2], [caption[j] for j in range(item*i, item*i+item)]) for i in range(val, total)]
    # train_ds = [(img, cap) for img, cap in train_dict.items()]
    # val_ds = [(img, cap) for img, cap in val_dict.items()]
    # test_ds = [(img, cap) for img, cap in test_dict.items()]
    return tf.data.experimental.from_list(train_ds), \
           tf.data.experimental.from_list(val_ds), \
           tf.data.experimental.from_list(test_ds)

def load_image(path, root_dir='./flickr30k-images/'):
    image = tf.io.decode_jpeg(tf.io.read_file(tf.constant(root_dir, dtype=tf.string) + path), channels=3)
    return tf.image.resize(image, IMAGE_SHAPE[:-1], method='lanczos5')

def mobileNetV3(choice='Large'):
    path = {'Large': './data/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5',
               'Small': './data/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5'}
    if choice == 'Large':
        mobilenet = keras.applications.MobileNetV3Large(
        input_shape=IMAGE_SHAPE, include_top=False, include_preprocessing=True, weights=path['Large'])
    else:
        mobilenet = keras.applications.MobileNetV3Small(
            input_shape=IMAGE_SHAPE, include_top=False, include_preprocessing=True, weights=path['Small'])
    mobilenet.trainable = False
    return mobilenet

def standardize(s):
    s = tf.strings.regex_replace(tf.strings.lower(s), f'[{re.escape(string.punctuation)}]', '')
    return tf.strings.join(['[START]', s, '[END]'], separator=' ')

@tf.autograph.experimental.do_not_convert
def tokenize_vocab(train_raw, vocab_size=5000, batch_size=1024):
    tokenizer = keras.layers.TextVectorization(
        max_tokens=vocab_size, standardize=standardize, ragged=True
    )
    tokenizer.adapt(train_raw.map(lambda fp,txt: txt).unbatch().batch(batch_size))
    # Create mappings for words to indices and indices to words.
    word2index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
    index2word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)
    return tokenizer, word2index, index2word

def match_shapes(images, captions):
    caption_shape = einops.parse_shape(captions, 'b c')
    captions = einops.rearrange(captions, 'b c -> (b c)')
    images = einops.repeat(images, 'b ... -> (b c) ...', c = caption_shape['c'])
    return images, captions

def prepare_txt(images, texts, tokenizer):
    tokens = tokenizer(texts)
    input_tokens = tokens[..., :-1]
    label_tokens = tokens[..., 1:]
    return (images, input_tokens), label_tokens

def prepare_dataset(ds, tokenizer, batch_size=32, shuffle_buffer=1024):
    # Load the images and batch them.
    ds = (ds.shuffle(5000).map(lambda path, caption: (load_image(path), caption))
          .apply(tf.data.experimental.ignore_errors()).batch(batch_size))

    def to_tensor(inputs, labels):
        (images, in_tok), out_tok = inputs, labels
        return (images, in_tok.to_tensor()), out_tok.to_tensor()

    prepare_text = lambda image, text: prepare_txt(image, text, tokenizer)
    return ds.map(match_shapes, AUTOTUNE).unbatch().shuffle(shuffle_buffer)\
        .batch(batch_size).map(prepare_text, AUTOTUNE).map(to_tensor, AUTOTUNE)

def save_dataset(ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
    # Load the images and batch them.
    ds = ds.map(lambda path, caption: (load_image(path), caption))\
        .apply(tf.data.experimental.ignore_errors()).batch(batch_size)

    # Run the feature extractor on each batch
    def gen():
        for (images, captions) in tqdm.tqdm(ds):
            feature_maps = image_model(images)

            feature_maps, captions = match_shapes(feature_maps, captions)
            yield feature_maps, captions

    # Wrap the generator in a new tf.data.Dataset.
    new_ds = tf.data.Dataset.from_generator(gen,
        output_signature=(
            tf.TensorSpec(shape=image_model.output_shape),
            tf.TensorSpec(shape=(None,), dtype=tf.string)))
    prepare_text = lambda image, text: prepare_txt(image, text, tokenizer)
    # Apply the tokenization
    new_ds = new_ds.map(prepare_text, AUTOTUNE).unbatch().shuffle(1000)

    # Save the dataset into shard files.
    def shard_func(i, item):
        return i % shards
    # substitutable for shard_func to be `lambda` expression
    new_ds.enumerate().save(save_path, shard_func=shard_func)

def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):
    def custom_reader_func(datasets):
        datasets = datasets.shuffle(1000)
        return datasets.interleave(lambda x: x, cycle_length=cycle_length)
    ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)
    # substitutable for `lambda` expression
    def drop_index(i, x):
        return x
    return ds.map(drop_index, AUTOTUNE).shuffle(shuffle).padded_batch(batch_size).prefetch(AUTOTUNE)

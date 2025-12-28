import tensorflow as tf


def _parse_tfrecord(binary_img=False, is_ccrop=False):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.float32)

        x_train = _transform_images(is_ccrop=is_ccrop)(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def _transform_images(is_ccrop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train

#修改这个函数
def load_tfrecord_dataset(dataset_path, batch_size, binary_img=False, is_ccrop=False):
    print(f"[*] 正在为 ArcFace 适配双输入加载: {dataset_path}")

    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=(112, 112),
        batch_size=batch_size,
        label_mode='int',
        shuffle=True  # 资深建议：训练集一定要打乱
    )

    # 1. 关键修改：让数据集无限循环，直到 model.fit 跑完指定的 epochs
    ds = ds.repeat()

    # 2. 核心逻辑：ArcFace 适配
    def format_train_data(x, y):
        x = (tf.cast(x, tf.float32) - 127.5) / 128.0
        return (x, y), y  # ( (图片, 标签), 真实标签 )

    ds = ds.map(format_train_data, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. 性能优化：预取数据，防止 GPU 等待 CPU
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def load_fake_dataset(size):
    """load fake dataset"""
    x_train = tf.image.decode_jpeg(
        open('./data/BruceLee.JPG', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)
    x_train = tf.image.resize(x_train, (size, size))

    labels = [0]
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
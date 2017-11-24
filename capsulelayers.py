"""
Deep Learning Türkiye Topluluğu için Merve Ayyüce Kızrak tarafından hazırlanmıştır. (http://www.ayyucekizrak.com/)

Bazı temel katmanlar (evrişimsel katmanlar) bir Kapsül Ağ oluşturmak için kullanılır.
Kapsül ağ modeli (CapsNet) oluşturmak için kullanılan katmanlar farklı veri setleri üzerinde de kullanılabilir,
sadece MNIST seti için tasarlanmamıştır.

"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    Vektör uzunluklarının hesaplanır. Bu, hata değerindeki (margin_loss) y_true ile aynı boyutta Tensor hesaplamak için kullanılır.
    Bu katmanı kullanarak modelin çıkışı ( `y_pred = np.argmax(model.predict(x), 1)` bu denklem kullanılarak )
    direkt olarak etiketleri kestirebilir.

    Girişler    : shape=[None, num_vectors, dim_vector]
    Çıkış       : shape=[None, num_vectors]

    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    shape=[None, num_capsule, dim_vector]  Bu Tensor filtre ya maksimum uzunluğuyla kapsül ya da ek bir giriş filtresidir.

    Maksimum uzunluklu kapsül hariç (yada belirtilen kapsül hariç), diğer tüm vektörler 0'a filtrelenir.
    Sonra filtrelenmiş tüm Tensörler düzgünleştirilir (flatten).

    Örneğin:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8,her bir iterasyonda "8" resim alınır.
                                                   her örnek 3 kapsül içerir  (dim_vector=2) vektörün uzunluğu 2'dir.
        y = keras.layers.Input(shape=[8, 3])  # Doğru etiketler. 8 örnek, 3 sınıf, (one-hot coding).
        out = Mask()(x)  # out.shape=[8, 6]
        # ya da
        out2 = Mask()([x, y])  # out2.shape=[8,6]. y'nin doğru etiketleri ile filtrelenir. Tabi ki y manipüle edilebilir.
        `
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # doğru etiket shape = [batch_size, n_classes], ile sağlanır.  (örneğin: one-hot code.)
            assert len(inputs) == 2
            inputs, mask = inputs
            mask = K.expand_dims(mask, -1)
        else:  # eğer doğru etiket yoksa, kapsüller maksimum uzunluklarıyla filtrelenir. Temel olarak kestirim için kullanılır.
            # kapsül uzunluğu hesaplanır.
            x = K.sqrt(K.sum(K.square(inputs), -1, True))
            # x aralığını max(new_x[i,:])=1 ve diğerleri << 0 yapmak için büyütür.
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            # x'teki bu maksimum değer 1 yapılır diğerleri 0 yapılır.
            # x'deki maksümum değer 1'e, diğerleri 0 olacak şekilde bölünür (clipped). Böylece `filtre (maske)` bir one-hot coding olur.
            mask = K.clip(x, 0, 1)

        return K.batch_flatten(inputs * mask)  # filtrelenmiş girişler, shape = [None, num_capsule * dim_capsule]

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # doğru değerler sağlanır
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # doğru olmayan değerler sağlanır
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    Kapsülde lineer olmayan aktivasyon kullanılır. Böylece büyük vektör 1'e küçük vektör 0'a yaklaşır.

    : "vectors" parametresi: bazı vektörler ezilir (squashed), N-dim tensor
    : "axis" parametresi: eksen ezilir (squash)
    : Fonksiyon Çıktısı: giriş vektörleri ile aynı uzunluklu bir Tensor
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    Kapsül Katmanı. Yoğun (Dense) katmanıyla benzerdir. Dense katmanı `in_num` girişlere sahiptir. Her biri skalerdır. önceki katmandan
    gelen nöron çıkıştır. Çıkış nöronları `out_num` ile gösterilir. Kapsül katmanı (CapsuleLayer) çıkış nöronlarının genişletilmiş
    skalar bir vektör halidir.
    Giriş: shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. Yoğun (Dense) katman için, input_dim_capsule = dim_capsule = 1.

    : "num_capsule" parametresi: her katmandaki kapsül sayısı
    : "dim_capsule" parametresi: ilgili katmandaki kapsülün çıkış vektörünün boyutu
    : "num_routing" parametresi: yönlendirme (routing) algoritmasının iterasyon sayısı
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "Giriş Tensorunun olması gereken boyutu shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Matris Dönüştürme
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # W değerleriyle çarpmaya hazırlamak için num_capsule boyutunu çoğaltır
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # 0 boyutta input_tiled taranarak `inputs * W` hesaplanır.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # İlk iki boyuta "batch" olarak bakılırsa;
        # sonra matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Başlangıç: Yönlendirme algoritması ---------------------------------------------------------------------#
        # İleri adım, `inputs_hat_stopped` = `inputs_hat`;
        # `inputs_hat_stopped`'dan `inputs_hat`'ya Geriye doğru, gradyan yayılımı olmaz.
        inputs_hat_stopped = K.stop_gradient(inputs_hat)

        # Birleştime katsayısı (coupling coefficient) başlangıçta 0'dır..
        # b.shape = [None, self.num_capsule, self.input_num_capsule]. değerine eşittir.
        # `b=K.zeros(shape=[batch_size, num_capsule, input_num_capsule])`. Sadece `batch_size` alınmıyor.
        b = K.stop_gradient(K.sum(K.zeros_like(inputs_hat), -1))

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # Son iterasyonda, gradyanı geriye yaymak için ,`inputs_hat` kullanarak `outputs` hesaplanır.
            if i == self.num_routing - 1:
                # c.shape =  [batch_size, num_capsule, input_num_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # İlk iki boyut, `batch` boyutudur.
                # sonra matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
                # outputs.shape=[None, num_capsule, dim_capsule]
                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [yok, 10, 16]
            else:  # Aksi halede 'b' yi güncellemek için `inputs_hat_stopped` kullanır. Bu yönde (yolda) herhangi bir granyan akışı olmaz.
                outputs = squash(K.batch_dot(c, inputs_hat_stopped, [2, 2]))

                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # İlk iki boyut, `batch` boyutudur.
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat_stopped, [2, 3])
        # Son: Yönlendirme algoritması -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Tüm kapsülleri birbirine bağlamak için Conv2D, `n_channels` kez uygulanır.

    : "inputs" parametresi: 4B Tensor, shape=[None, width, height, channels]
    : "dim_capsule" parametresi: Çıkış kapsül vektörlerinin boyutu
    : "n_channels" parametresi: Kapsül tiplerinin sayısı
    : Fonksiyon Çıktısı: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

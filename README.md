**Prof. Geoffrey Hinton**’ın “**Dynamic Routing Between Capsules**” makalesindeki **CapsNet** algoritmasının **Keras** Uygulamasıdır.

Deep Learning Türkiye Topluluğu için [Merve Ayyüce Kızrak](http://www.ayyucekizrak.com/) tarafından hazırlanmıştır.

**Kapsül Ağları** hakkında detaylı bilgi için: [Yapay Zekada Büyük Yenilik: Kapsül Ağları (Capsule Networks)](http://www.ayyucekizrak.com/blogdetay/yapay-zeka-ve-derin-ogrenme-yazi-dizisi/yapay-zekada-buyuk-yenilik-kapsul-aglari-capsule-networks)

Geçerli sürümü TensorFlow’da hazırlanmıştır. Keras sürümünden farklı sürümlere kolaylıkla dönüştürülüp yeniden yazılabilir.

* Amaç: Kapsül ağının el yazısı rakamları tanımaktaki performansını değerlendirmek.
* Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)
* Algoritma: Kapsül Ağları (Capsule Networks-CapsNet)

## Adım Adım Kullanım

Terminali açıp

```git clone https://github.com/deeplearningturkiye/kapsul-agi-capsule-network.git```
ile repoyu bilgisayarınıza indirin.

```cd kapsul-agi-capsule-network```

ile klasörü açın.

```python capsulenet.py```

ile çalıştırın.

    
## Validasyon Başarımı (Sadece Artış Gösterenler)

| Epoch         | Validasyon Doğruluk Oranı|
| ------------- |:-----------------------: |
|        1      | 98.58                    |
|        2      | 98.96                    |
|        3      | 99.31                    |
|        6      | 99.38                    |
|        10     | 99.41                    |
|        11     | 99.50                    |
|        12     | 99.56                    |
|        16     | 99.63                    |
|        34     | 99.66                    |

## Test Başarımı 

* **10 bin test verisi** ile test işlemi gerçekleştirildiğinde **%99.61 doğruluk oranı** elde edilmiştir. 
* Model **%0.39 hata** ile çalışmaktadır.

## 50 Epoch Çalışma Süresi

| Süre          | Bilgisayar Konfigürasyonu                                            |
| ------------- |:-----------------------:                                             |
|  68 saat      | İşlemci: Intel(R) Core (TM) i5-337U CPU @ 1.8 Ghz 1.8 Ghz, RAM: 4 GB |


## Kaynaklar
* https://arxiv.org/pdf/1710.09829.pdf (Dynamic Routing Between Capsules)
* https://github.com/XifengGuo/CapsNet-Keras
* http://www.ayyucekizrak.com/blogdetay/yapay-zeka-ve-derin-ogrenme-yazi-dizisi/yapay-zekada-buyuk-yenilik-kapsul-aglari-capsule-networks

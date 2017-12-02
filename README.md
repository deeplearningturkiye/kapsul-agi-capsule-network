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
    
## Test Başarımı 

* **10 bin test verisi** ile test işlemi gerçekleştirildiğinde **MNIST** veriseti için **%99.61 doğruluk oranı** ve **FASHION MNIST** veriseti için **%99.22 doğruluk oranı** elde edilmiştir. 

## MNIST için 50 Epoch Çalışma Süresi

| Süre          | Bilgisayar Konfigürasyonu                                            |
| ------------- |:-----------------------:                                             |
|  68 saat      | İşlemci: Intel(R) Core (TM) i5-337U CPU @ 1.8 Ghz 1.8 Ghz, RAM: 4 GB |


## Kaynaklar
* https://arxiv.org/pdf/1710.09829.pdf (Dynamic Routing Between Capsules)
* https://github.com/XifengGuo/CapsNet-Keras
* http://www.ayyucekizrak.com/blogdetay/yapay-zeka-ve-derin-ogrenme-yazi-dizisi/yapay-zekada-buyuk-yenilik-kapsul-aglari-capsule-networks

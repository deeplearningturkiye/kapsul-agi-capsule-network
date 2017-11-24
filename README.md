Prof. Geoffrey Hinton’ın “Dynamic Routing Between Capsules” makalesindeki CapsNet algoritmasının Keras Uygulamasıdır.

Deep Learning Türkiye Topluluğu için Merve Ayyüce Kızrak tarafından hazırlanmıştır. (http://www.ayyucekizrak.com/)

Geçerli sürümü TensorFlow’da hazırlanmıştır. Keras sürümünden farklı sürümlere kolaylıkla dönüştürülüp yeniden yazılabilir.

* Amaç: Kapsül ağının el yazısı rakamları tanımaktaki performansını değerlendirmek.
* Kaynak: https://arxiv.org/pdf/1710.09829.pdf (Dynamic Routing Between Capsule)
* Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)
* Algoritma: Kapsül Ağları (Capsule Networks-CapsNet)

## Adım Adım Kullanım

Terminali açın.

```git clone https://github.com/deeplearningturkiye/kapsul-agi-capsule-network.git```

ile repoyu bilgisayarınıza indirin.

```cd kapsul-agi-capsule-network```

ile klasörü açın.

```python capsulenet.py```

ile çalıştırın.

    
## Validasyon Başarımı
* 1. epoch sonrasında %98.5'e
* 20. epoch sonrasında %99.5'e 
* 50. epoch sonrasında %99.66’ya 

yükselmektedir.

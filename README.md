# BLM0362_NaiveBayes

# Denetimli öğrenme sınıflandırma algoritmaları: Naive-Bayes nedir,çalışma mantığı nedir, kullanım alanları, örnekleri

# Denetimli Öğrenim Nedir? 

"Denetimli öğrenim" terimi, bir yapay zeka algoritmasının veri kümesi üzerinde çalışırken, önceden eğitilmiş bir modele dayanarak ve bu modelden alınan yanıtları kullanarak öğrenme sürecini ifade eder. Bu algoritma, veri kümesindeki örneklerin etiketleri veya doğru cevapları bilindiğinde çalışır ve bu bilgileri kullanarak modele öğretilir. Böylece, algoritma, veri kümesinde bulunan yeni örnekleri sınıflandırmak veya tahmin etmek için kullanılabilir.

Denetimli öğrenim algoritması, veri kümesinde bulunan örneklerin etiketleri veya doğru cevapları bilinen bir yapay zeka algoritmasıdır. Bu algoritma, bu bilgileri kullanarak bir model eğitir ve daha sonra bu modele dayanarak veri kümesinde bulunan yeni örnekleri sınıflandırma veya tahmin etme işlemlerini yapar.

Örneğin, bir denetimli öğrenme algoritması, resimlerdeki nesneleri tanımlamak için eğitilebilir. Bu algoritma, resimlerdeki nesnelerin etiketlerini veya doğru cevaplarını bilir ve bu bilgileri kullanarak bir model eğitir. Daha sonra bu modele dayanarak, yeni bir resim görüntülendiğinde, algoritma bu resimdeki nesneyi doğru bir şekilde tanımlayabilir.

<p align="center"><img width="500"  src="Images/denetimli.png">

Şekildeki örnekte Hexagon, Square, Triangle olarak etiketlenmiş veriler görülmektedir. İlk olarak bu şekilde verilerin bulunduğu bir eğitim seti ile bu model eğitilir. Eğitilen model, test veri seti ile de test edilerek, test veri setinde bulunan şekilleri doğru olarak tahmin etmesi beklenir.

Denetimli öğrenim algoritmaları;

1. Naive Bayes: Bu algoritma, veri kümesinde bulunan örneklerin sınıflandırılması veya tahmin edilmesi işlemlerini yapmak için kullanılan bir yapay zeka algoritmasıdır. Naive Bayes, veri kümesinde bulunan örneklerin etiketlerini veya doğru cevaplarını kullanarak bir model eğitir ve daha sonra bu modele dayanarak, veri kümesinde bulunan yeni örnekleri sınıflandırma veya tahmin etme işlemlerini yapar. Naive Bayes, veri kümesinde bulunan örneklerin özelliklerinin birbirlerinden bağımsız olduğu varsayımına dayanarak çalışır.
2. Nöral Ağlar: Bu algoritma, veri kümesinde bulunan örneklerin sınıflandırılması veya tahmin edilmesi işlemlerini yapmak için kullanılan bir yapay zeka algoritmasıdır. Nöral ağlar, insan beyninin yapısına benzer bir yapıya sahiptir ve bu yapı sayesinde, veri kümesinde bulunan örneklerin sınıflandırılması veya tahmin edilmesi işlemlerini gerçekleştirirler. Nöral ağlar, veri kümesinde bulunan örneklerin etiketlerini veya doğru cevaplarını kullanarak bir model eğitir ve daha sonra bu modele dayanarak, veri kümesinde bulunan yeni örnekleri sınıflandırma veya tahmin etme işlemlerini yapar.
3. Karar Ağaçları: Karar ağaçları, verilerin birleştirilmesi ve dallanması yoluyla bir karar verme mekanizması oluşturmayı amaçlayan bir denetimli öğrenme algoritmasıdır. Örneğin, bir karar ağacı ev sahipliği yapılacak bir parti için en uygun zamanı seçmek için kullanılabilir. Girdi olarak mevsim, hava durumu ve parti için öngörülen katılım sayısı gibi değişkenler verilebilir ve algoritma bu değişkenleri kullanarak en uygun zamanı belirleyebilir.
4. Destek Vektör Makineleri (SVM): Destek vektör makineleri, verileri bir hiperdüzlemde ayrıştıran ve bu ayrışma sonucunda bir sınıflandırma gerçekleştiren bir denetimli öğrenme algoritmasıdır. Örneğin, bir destek vektör makinesi ev sahipliği yapılacak bir parti için en uygun zamanı seçmek için kullanılabilir. Algoritma bu verilere dayanarak ev sahipliği yapılacak zamanı sınıflandırabilir.
5. Rastgele Orman: Rastgele orman, bir grup karar ağacının bir araya gelmesiyle oluşturulan bir denetimli öğrenme algoritmasıdır. Bu algoritma, her bir karar ağacının verilere farklı açılardan bakmasını sağlar ve bu sayede doğruluğu artırılmış bir tahmin yapabilir.
6. Lineer Regresyon: Bu algoritma, veri kümesinde bulunan örneklerin birbirleriyle ilişkisini bulmak ve bu ilişkiyi kullanarak, veri kümesinde bulunan yeni örneklerin tahmin edilmesi işlemlerini yapmak için kullanılan bir yapay zeka algoritmasıdır. Lineer regresyon, veri kümesinde bulunan örneklerin etiketlerini veya doğru cevaplarını kullanarak bir model eğitir ve daha sonra bu modele dayanarak, veri kümesinde bulunan yeni örneklerin tahmin edilmesi işlemini yapar. Lineer regresyon, örnekler arasındaki ilişkiyi bir doğruya yakın bir şekilde ifade etmeyi hedefler.
7. Logistic Regresyon: Bu algoritma, veri kümesinde bulunan örneklerin sınıflandırılması veya tahmin edilmesi işlemlerini yapmak için kullanılan bir yapay zeka algoritmasıdır. Logistic regresyon, veri kümesinde bulunan örneklerin etiketlerini veya doğru cevaplarını kullanarak bir model eğitir ve daha sonra bu modele dayanarak, veri kümesinde bulunan yeni örnekleri sınıflandırma veya tahmin etme işlemlerini yapar. Logistic regresyon, örneklerin sınıflandırılması veya tahmin edilmesi işlemini, örneklerin birbirleriyle olan ilişkisini bir logaritmik doğruya yakın bir şekilde ifade ederek gerçekleştirir.

Denetimli öğrenim algoritmalarının kullanım alanlarına dair günlük hayattan örnekler;

- Spam e-posta filtreleme: Bir e-posta istemcisi (Outlook, Gmail vb.), gelen e-posta mesajlarını otomatik olarak spam veya değil spam olarak sınıflandırır. Bu sınıflandırma, denetimli öğrenim algoritmaları kullanılarak yapılır.
- Öneri sistemleri: Bir e-ticaret sitesi (Amazon, eBay vb.), kullanıcıların daha önce satın aldıkları veya inceledikleri ürünlerden yola çıkarak, kullanıcılara ürün önerileri yapar. Bu öneriler, denetimli öğrenim algoritmaları kullanılarak hesaplanır.
- Sağlık takipleri: Bir sağlık uygulaması (Fitbit, Apple Health vb.), kullanıcının günlük aktivitelerini (adım sayısı, kalori harcaması vb.) takip eder ve bu veriler kullanılarak kullanıcının sağlık durumunun tahmin edilmesine yardımcı olur. Bu tahminler, denetimli öğrenim algoritmaları kullanılarak yapılır.
- Ses tanıma: Bir ses tanıma uygulaması (Google Assistant, Siri vb.), kullanıcının söylediği kelimeleri tanıyarak bu kelimeleri metne dönüştürür. Bu tanıma, denetimli öğrenim algoritmaları kullanılarak yapılır.
- Görüntü tanıma: Bir görüntü tanıma uygulaması (Google Görüntü Tanıma, Microsoft Cognitive Services vb.), verilen bir görüntüde bulunan nesneleri tanıyarak bu nesneleri adlandırır. Bu tanıma, denetimli öğrenim algoritmaları kullanılarak yapılır.
- Yüz tanıma: Bir yüz tanıma uygulaması (Facebook, Google Photos vb.), verilen bir görüntüde bulunan yüzleri tanıyarak bu yüzleri kişilere eşleştirir. Bu tanıma, denetimli öğrenim algoritmaları kullanılarak yapılır.

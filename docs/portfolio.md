# KANAL | Klasifikasi berita Indonesia dari RSS

KANAL mengklasifikasikan judul dan ringkasan berita Indonesia ke delapan kanal.
Pada 517 artikel uji, model TF-IDF + LinearSVC mencapai **macro-F1 0,420**,
dibandingkan **0,026** untuk baseline kelas mayoritas.

## Masalah dan pendekatan

Empat penerbit memakai rubrik yang berbeda. Pipeline mengambil berita dari RSS,
menyeragamkan label di DuckDB dan dbt, lalu menyimpan snapshot 1.295 artikel
sebagai Parquet. [Dataset publik](https://huggingface.co/datasets/aeroo11/kanal-berita)

- **Evaluasi temporal.** Artikel uji berasal dari jendela satu hari setelah
  batas validasi. Artikel yang hampir duplikat dikelompokkan agar satu kelompok
  tetap berada dalam satu split.
- **Promosi berbasis hasil uji.** Model dibandingkan dengan baseline pada split
  yang sama. Pipeline menyimpan konfigurasi, metrik per kelas, dan alasan
  promosi agar keputusan bisa diperiksa ulang.
- **Model berversi di API.** FastAPI menyertakan ID model di respons.
  Registry menyimpan versi sebelumnya agar rollback bisa diuji secara lokal.

## Hasil dan batasannya

| Model | Macro-F1 | Akurasi |
| --- | ---: | ---: |
| Baseline kelas mayoritas | 0,026 | 11,4% |
| TF-IDF + LinearSVC | 0,420 | 48,4% |

F1 `olahraga` mencapai 0,839, sementara F1 `ekonomi` hanya 0,144. Kelas
`hukum-kriminal` memiliki satu artikel uji. Model mengenalinya, tetapi juga
memberi label itu pada 22 artikel dari kanal lain. Hasil tersebut menunjukkan
perlunya data tambahan untuk mengevaluasi kelas yang jarang muncul.
[Protokol dan analisis evaluasi](evaluation.md#v1-results)

## Coba demo

Dengan Docker aktif, jalankan image yang sudah berisi model hasil evaluasi.

```sh
docker run --rm -p 127.0.0.1:7860:7860 aeroofan/kanal-berita:v1-d15a999
```

Buka [Swagger lokal](http://127.0.0.1:7860/docs), pilih `POST /predict`, lalu
klik **Try it out**. Kirim contoh berikut.

```json
{"articles":[{"title":"Persib melaju ke final setelah menang lewat adu penalti"}]}
```

Model `0f604fff6e957463` memprediksi `olahraga` dengan probabilitas sekitar
0,293. Respons juga mencantumkan probabilitas kanal lain. [Image Docker](https://hub.docker.com/r/aeroofan/kanal-berita/tags)

[Reproduksi pelatihan dan evaluasi](../README.md#coba-demo-dari-checkout-bersih)

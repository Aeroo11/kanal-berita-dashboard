# Sumber data, lisensi, dan kebijakan polling

Proyek ini membaca RSS feed publik dari tiga penerbit berita Indonesia. Dokumen
ini menyatakan persis apa yang dikumpulkan, apa yang tidak, dan bagaimana
pengumpulannya berperilaku — karena proyek riset yang diam-diam mengambil lebih
dari haknya memang pantas diblokir.

## Sumber

| Penerbit | Feed | Indeks RSS |
|---|---|---|
| ANTARA (kantor berita negara) | 11 | `https://www.antaranews.com/rss/` |
| CNN Indonesia | 7 | `https://www.cnnindonesia.com/{section}/rss` |
| Liputan6 | 7 | `https://feed.liputan6.com/rss/{channel}` |

## Yang disimpan

Hanya apa yang penerbit sendiri tempatkan di dalam feed sindikasi:

- judul
- ringkasan atau deskripsi dari feed
- canonical URL artikel (tautan balik ke sumber)
- penerbit dan channel, sebagai *provenance*
- waktu terbit
- waktu pengambilan dan identitas feed

## Yang sengaja tidak disimpan

- **Isi artikel.** Tidak ada halaman di luar feed yang pernah diminta. Tidak ada
  scraper di repository ini, dan menambahkannya secara eksplisit di luar
  cakupan.
- **Gambar.** Markup thumbnail yang menempel di ringkasan dibuang saat ingest.
- **Apa pun di balik paywall, login, atau larangan `robots.txt`.**

Teks artikel tetap milik penerbitnya. Repository ini menyimpan metadata hasil
sindikasi dan menautkan balik ke aslinya. Tidak ada di sini yang bisa
menggantikan membaca sumbernya, dan tidak ada teks utuh yang diterbitkan ulang.

## Kebijakan polling

RSS memang ada untuk dikonsumsi, dan cara tetap diterima adalah bersikap seperti
konsumen yang tahu aturan:

| Perilaku | Nilai |
|---|---|
| Interval polling | tiap jam, per feed |
| Konkurensi | **satu request pada satu waktu** — tidak pernah paralel |
| Jeda antar request | 1 detik |
| Conditional request | `If-None-Match` / `If-Modified-Since` pada tiap polling |
| Timeout | 20 detik |
| Percobaan ulang | 3 kali, exponential backoff, `Retry-After` dihormati |
| Status yang diulang | hanya 408, 425, 429, 5xx |
| Yang tidak diulang | 403, 404 — ini putusan, mengulanginya cuma bising |
| Circuit breaker | 3 kegagalan berturut-turut → feed dilewati 1 jam |
| User-Agent | deskriptif, memuat jalur kontak |

`User-Agent` yang dipakai saat ini:

```
kanal-research/0.1 (+https://github.com/Aeroo11/kanal-berita-dashboard)
Indonesian news classification research; contact via GitHub issues
```

Conditional request lebih penting daripada kelihatannya: feed yang di-polling
tiap jam sebagian besar waktu tidak berubah, sehingga penerbit cukup menjawab
`304` tanpa body. Itu lebih hemat untuk mereka daripada untuk kami.

## Kenapa tiap jam, dan kenapa tidak bisa lebih jarang

RSS feed adalah *sliding window* berisi 25–100 item terakhirnya. Tidak ada
endpoint arsip dan tidak ada parameter `?since=`. **Jam yang tidak tertangkap
tidak bisa dikembalikan.** Polling tiap jam adalah laju minimum yang menjaga
jendela itu tidak menyalip kami di hari yang ramai berita; ini bukan usaha untuk
menjadi yang pertama atas apa pun.

## Label

Label sebuah artikel adalah channel tempat ia disindikasikan — artikel dari
`antaranews.com/rss/ekonomi.xml` berlabel `ekonomi`. Channel penerbit dipetakan
ke delapan kelas kanonik di `src/kanal/ingest/sources.py`, dan setiap keputusan
yang bersifat pertimbangan dicatat beserta alasannya di sana.

Label-label ini adalah keputusan redaksi penerbit, bukan kebenaran mutlak.
Antar penerbit pun bisa berbeda untuk berita hasil sindikasi yang sama, dan
perbedaan itu nanti akan **diukur**, bukan diasumsikan hilang — lihat rencana
pengukuran plafon *label noise* di tahap evaluasi.

## Kalau Anda penerbit

Kalau Anda lebih suka proyek ini tidak mem-polling feed Anda, silakan buka
*issue* di repository dan sumbernya akan dihapus. Tidak perlu penjelasan
tambahan.

## Redistribusi

Dataset turunan yang nanti diterbitkan ke Hugging Face berisi judul, ringkasan,
tautan, dan label — metadata sindikasi yang sama seperti dijelaskan di atas.
Ditujukan untuk riset, dan bukan pengganti konten milik penerbit.

Korpus awal yang dipakai untuk uji coba *cold-start* punya lisensinya sendiri:

- `indonesian-nlp/id_newspapers_2018` — CC-BY-4.0
- `fahadh4ilyas/indonesian_news_datasets` — CC-BY-NC-4.0 (non-komersial)

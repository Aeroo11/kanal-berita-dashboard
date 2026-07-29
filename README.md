# KANAL

Platform *model lifecycle* untuk klasifikasi berita Indonesia: **cari model
terbaik, deploy, dan jaga supaya tetap jujur.**

> **Status: Stage 1 berjalan — ingestion, warehouse, dan dbt.**
> README ini hanya menjelaskan apa yang sudah benar-benar dibangun. Bagian baru
> ditambahkan setelah kodenya ada, bukan sebelumnya.

---

## Yang sudah jalan hari ini

Ingestion yang mem-*polling* 25 RSS feed dari tiga penerbit Indonesia tiap jam,
warehouse DuckDB, dan model dbt beserta *data contract*-nya.

```bash
uv sync
uv run kanal status      # isi feed registry dan landing zone
uv run kanal ingest      # satu siklus polling
uv run kanal ingest      # jalankan lagi — nol baris baru
uv run kanal load        # landing zone → DuckDB

cd dbt && DBT_PROFILES_DIR=. uv run dbt build
```

Siklus kedua yang mendaratkan **nol baris** itu bukan kebetulan — lihat bagian
*Idempotency*.

Terukur pada siklus pertama yang sungguhan: **1.270 artikel, 25/25 feed HTTP
200, 27 detik**. `dbt build`: **46 node, semua lolos** — 1 seed, 3 view,
2 tabel, 40 test.

---

## Masalah yang dikerjakan

Mengklasifikasikan artikel berita Indonesia ke salah satu dari delapan *kanal* —
`politik`, `ekonomi`, `olahraga`, `teknologi`, `hiburan`, `internasional`,
`hukum-kriminal`, `gaya-hidup-kesehatan` — **hanya dari judul dan ringkasan
RSS**, tidak pernah dari isi artikel.

Judul-saja adalah tingkat kesulitan yang dipilih sengaja. Sepuluh sampai
dua puluh lima token itu cukup sedikit untuk membuat tugasnya benar-benar sulit,
sehingga tersisa ruang untuk perbandingan yang sebenarnya jadi inti proyek ini:
model TF-IDF seharga pecahan sen melawan LLM yang empat orde besaran lebih
mahal — dan pertanyaan kapan selisih itu layak dibayar.

Konsekuensi lain: tidak ada *scraping* isi artikel. Tidak ada yang disimpan
melebihi apa yang penerbit sendiri pilih untuk disebarkan lewat feed.

---

## Kenapa RSS, dan dari mana label-nya datang

Label sebuah artikel adalah **section tempat penerbitnya mengarsipkan artikel
itu**. Artikel yang muncul di `antaranews.com/rss/ekonomi.xml` berlabel
`ekonomi`.

Satu keputusan itu yang membuat sisa proyek ini mungkin. Label datang otomatis,
terus-menerus, dan gratis — sehingga *retraining* nanti punya sesuatu yang baru
untuk dipelajari, bukan sekadar cron job yang mengulang *fitting* dataset beku
lalu menyebut dirinya MLOps.

### Sumber

| Penerbit | Feed | Section di URL artikel | `<category>` per item |
|---|---|---|---|
| **ANTARA** (kantor berita negara) | 11 | tidak | tidak |
| **CNN Indonesia** | 7 | **ya** | tidak |
| **Liputan6** | 7 | **ya** | **ya** |

Perbedaan itu bukan gangguan — itu justru eksperimennya. Lihat *Kebocoran
label*.

---

## Catatan engineering

### Idempotency, di atas sumber yang tidak bisa di-backfill

RSS feed adalah *sliding window* berisi 25–100 item terakhir. Tidak ada endpoint
arsip dan tidak ada parameter `?since=`: **jam yang tidak tertangkap hilang
permanen.** Satu fakta itu menentukan sebagian besar desainnya.

Polling tiap jam berarti melihat artikel yang sama berulang kali. Jadi identitas
sebuah artikel adalah hash dari URL yang sudah di-*canonicalise*, bukan URL
mentahnya — artikel yang sama datang sebagai

```
https://www.antaranews.com/berita/123/judul?utm_source=rss
https://www.antaranews.com/berita/123/judul/
http://antaranews.com/berita/123/judul#top
```

dan ketiganya harus menyusut jadi satu `article_key`. Canonicalisation memaksa
https, membuang `www.` dan port default, menghapus parameter tracking, mengurut
sisanya, membuang fragment, dan menormalkan trailing slash — tapi **sengaja
tidak** me-*lowercase* path, karena sebagian CMS memang menyajikan slug yang
case-sensitive dan melipatnya akan menggabungkan artikel yang berbeda.

Sebelum menulis, satu siklus membaca kembali key apa saja yang sudah ada di
partisi tujuannya, lalu hanya menulis yang baru. Menjalankan ulang sebuah
polling karena itu jadi *no-op*. Pembacaan itu dari disk, bukan dari set di
memori — karena restart proses justru saat idempotency paling dibutuhkan.

### Jadi tamu yang tahu diri

Feed diterbitkan untuk disebarkan, dan cara tetap diterima adalah bersikap
seperti konsumen yang tahu aturan:

- *conditional request* lewat `ETag` / `If-Modified-Since`, sehingga feed yang
  tidak berubah cukup dijawab `304` tanpa body,
- satu request pada satu waktu dengan jeda di antaranya — tidak pernah paralel,
- `User-Agent` deskriptif yang memuat jalur kontak,
- *exponential backoff* yang menghormati `Retry-After`,
- yang disimpan hanya judul, ringkasan, canonical URL, penerbit, dan timestamp —
  **tidak pernah isi artikel**, tidak pernah menerbitkan ulang teksnya.

Selengkapnya di [`DATA.md`](DATA.md).

### Satu sumber bermasalah tidak boleh menjatuhkan yang lain

Dua penerbit diketahui tidak stabil — Tempo membalas `403` untuk User-Agent yang
tidak dikenal, Detik memutus koneksi secara intermiten. Siklus yang mati di
kegagalan pertama ikut kehilangan feed yang sehat, dan jam itu tidak bisa
dikembalikan.

Jadi setiap feed diisolasi, dan sumber yang gagal berulang kali akan memicu
*circuit breaker* lalu dilewati selama masa *cooldown*, bukan dicoba ulang tiap
siklus. Kegagalan juga diklasifikasikan, tidak disamaratakan: `408/429/5xx`
bersifat sementara dan layak diulang, sedangkan `403` dan `404` adalah
**putusan** tentang siapa kita atau apa yang kita minta — mengulanginya cuma
menambah bising.

Satu siklus keluar dengan status non-zero kalau ada penerbit yang sama sekali
tidak memberi respons berguna, sehingga *scheduled run* menyala merah alih-alih
sukses diam-diam dengan lubang di datanya. Satu channel yang sepi itu wajar;
satu penerbit yang mati total tidak.

### Kebocoran label, yang di sini nyata dan terukur

Label **adalah** asal-usul feed, jadi ia bocor lewat beberapa jalur sekaligus.
Diukur pada 1.270 baris:

| Sumber | URL yang membocorkan label | Lewat nama channel | Lewat label kanonik |
|---|---|---|---|
| **CNN** | **100,0%** | 100,0% | 85,7% |
| **Liputan6** | **98,9%** | 98,9% | 1,4% |
| ANTARA | 4,1% | 3,6% | 2,3% — kebetulan kata, bukan struktur |

**83,1% dari seluruh baris punya URL yang membocorkan jawabannya.** Hanya 215
baris yang bersih, dan 211 di antaranya dari ANTARA.

Angka Liputan6 itu adalah koreksi. Versi pertama pengukuran ini melaporkannya
**1,4%**, karena hanya mengecek apakah URL memuat label *kanonik*. Liputan6
memakai nama channel-nya sendiri — path-nya `/bisnis/read/…` sementara labelnya
`ekonomi`. Padahal channel memetakan ke label secara deterministik, jadi
membocorkan channel sama tuntasnya dengan membocorkan label.

Setelah dikoreksi, temuannya justru lebih tajam daripada klaim awalnya: bukan
"CNN bocor", melainkan **ANTARA adalah satu-satunya sumber yang tidak bocor.**
Itulah yang menjadikannya kelompok kontrol — selisih F1 antara model ANTARA-saja
dan model semua-sumber **mengukur** berapa banyak yang dihadiahkan kebocoran.

Model apa pun yang melihat URL, feed id, atau category akan mendapat akurasi
~100% dan tidak mempelajari apa pun. Field-field itu disimpan untuk *provenance*
dan audit, dan dikecualikan dari fitur secara struktural.

Boilerplate penerbit adalah versi yang lebih halus dari masalah yang sama.
`"Liputan6.com, Jakarta - "` di awal ringkasan mengidentifikasi penerbitnya, dan
tiap penerbit punya komposisi kanal yang berbeda — jadi prefiks itu sendiri
sudah setengah label. Dibuang saat ingest.

### Konten evergreen, dan kenapa ini mengubah temporal split

Feed ANTARA mencampur *explainer* abadi ke dalam berita — profil, artikel
"mengenal…", daftar jadwal, yang bertahan di feed tanpa batas waktu. Terukur:

| Sumber | Item berumur >30 hari |
|---|---|
| ANTARA | **141 / 220 (64%)** — sebagian hampir setahun |
| Liputan6 | 42 / 350 |
| CNN | **0 / 700** |

Ini properti sumbernya, bukan bug parsing, dan dampaknya serius. *Temporal
split* naif (train ≤ T−14 hari, test > T−7 hari) akan mendorong hampir seluruh
baris ANTARA ke train dan meninggalkan test set yang **didominasi CNN — sumber
paling bocor**. Evaluasinya akan tampak wajar dan tidak berarti apa-apa.

Split-nya harus dibangun dengan tahu ini, jadi warehouse harus bisa
menyatakannya: query umur per sumber dikunci di `tests/test_loader.py`. Timestamp
yang tidak terbaca disimpan `null`, bukan ditebak — waktu terbit yang salah akan
merusak justru split yang jadi fondasi seluruh evaluasi.

### Setiap default DuckDB adalah default *host*

`temp_directory` defaultnya path relatif `.tmp` yang di-resolve terhadap working
directory; `memory_limit` sekitar 80% RAM yang dilaporkan mesin; `threads`
sebanyak core host. Di laptop ini terbaca 25 GiB dan 20 thread. Di dalam CI
runner atau container, default yang sama berarti "tulis spill ke direktori
read-only, pesan memori yang tidak kamu punya, dan jalankan thread sepuluh kali
lebih banyak dari core yang ada".

Semuanya dipin di `warehouse/duck.py`, dan *extension autoload* dimatikan —
tidak ada yang memakainya, dan kalau dibiarkan menyala DuckDB akan mencoba
mengunduh lalu menulis ke `$HOME` saat query berjalan.

### Satu writer, banyak reader

Satu koneksi DuckDB mengeksekusi satu statement pada satu waktu. Membagi satu
koneksi ke beberapa reader yang bersamaan akan mengantrekan semuanya di belakang
query paling lambat, dan langsung *deadlock* begitu ada orchestrator. Jadi:
tepat satu writer, dan reader mendapat koneksi pendek masing-masing. `writer()`
dan `reader()` ada supaya batasan itu terlihat di kode, bukan ditemukan saat
runtime.

### Loader yang sengaja bodoh

Loader membaca JSONL, melakukan *anti-join* pada `article_key`, memasukkan yang
baru, dan mencatat file yang sudah dikonsumsi. Tidak ada pembersihan, tidak ada
pembentukan ulang, tidak ada penafsiran — semuanya jatah dbt, tempat semua itu
ter-*version*, ter-*test*, dan terlihat sebagai lineage. Loader yang diam-diam
mentransformasi adalah loader yang keluarannya tidak bisa dijelaskan siapa pun.

Dua properti yang wajib dimilikinya, keduanya di-test:

- **Idempotent** — memuat file yang sama dua kali tidak menambah apa pun.
  Anti-join juga melipat satu berita kantor berita yang mendarat di partisi dua
  sumber menjadi satu baris.
- **Incremental** — file yang sudah tercatat di `_load_log` tidak dibuka lagi,
  sehingga proses load tetap murah saat landing zone tumbuh melewati ratusan
  ribu baris.

### Data contract: yang dipilih adalah yang gagal diam-diam

Test yang berguna bukan yang paling banyak, tapi yang menangkap kegagalan tanpa
gejala. Empat *singular test* di `dbt/tests/` masing-masing ada karena satu
skenario spesifik:

| Contract | Kegagalan yang dicegah |
|---|---|
| `taxonomy_coverage` | Penerbit menambah section baru → artikel mengalir dengan label yang tidak pernah ditinjau siapa pun. Kerusakannya di **label**, tempat terburuk untuk kegagalan senyap |
| `taxonomy_agrees_with_ingestion` | `sources.py` dan `taxonomy_map.csv` adalah dua ekspresi dari satu keputusan — dan dua salinan satu kebenaran pasti melenceng |
| `no_future_publication` | Timestamp yang di-*post-date* akan tersortir ke masa depan dan masuk test set terlepas dari kapan artikelnya ditulis. Itu definisi *lookahead* |
| `every_source_reported_today` | Penerbit menghilang. **Warn, bukan error** — penerbit yang diblokir itu keputusan yang harus diambil, bukan alasan menghentikan pemrosesan sumber yang sehat |

Ditambah *source freshness*: warn di 2 jam, error di 6 jam. Ingestion jalan tiap
jam, dan siklus yang terlewat tidak bisa di-backfill.

Seed taksonomi membawa kolom `confidence` dan `notes` per channel, sehingga
keputusan yang bersifat pertimbangan tertulis di tempat yang bisa dibantah. Ini
sudah terbayar: **semua label `politik` kecuali 20 berasal dari mapping
ber-confidence medium** (`nasional` milik CNN), yang memprediksi persis dari mana
kebingungan politik/hukum-kriminal nanti akan datang.

### Fixture CI harus mereproduksi masalahnya, bukan cuma skemanya

CI tidak boleh menyentuh penerbit, jadi `dbt build` di CI berjalan di atas
fixture sintetis. Fixture yang sekadar berbentuk benar akan lolos semua test
sambil tidak membuktikan apa pun — jadi fixture-nya sengaja mereproduksi
properti yang jadi perhatian model: kebocoran URL CNN, item evergreen ANTARA,
satu berita kawat yang difile di bawah dua section berbeda, dan satu timestamp
yang tidak terbaca.

Versi pertamanya justru salah dengan cara yang instruktif. Judul sintetisnya
diberi sufiks per **sumber**, bukan per feed, sehingga keempat feed ANTARA
memancarkan judul identik — membentuk cluster duplikat palsu berukuran empat dan
memproduksi **102 baris "label disagreement" yang seluruhnya artefak**. Model-nya
benar; fixture-nya yang berbohong. Sekarang: tepat satu cluster lintas-sumber,
yaitu berita kawat yang memang ditanam.

### Evolusi skema

Feed berubah bentuk tanpa pemberitahuan. Liputan6 mengirim `<category>`, ANTARA
tidak; format tanggal berbeda-beda; bulan depan bisa saja ada yang mulai
mengirim `media:content`.

Kalau parsing hanya menyimpan field yang dipahami hari ini, hari saat sebuah
feed berubah adalah hari yang datanya hilang diam-diam. Sebagai gantinya, raw
layer menyimpan **setiap** nilai skalar yang dikirim penerbit di dalam objek
`extra` beserta `schema_version`, menurunkan apa yang bisa diturunkan, dan
mencatat kegagalan parsing per item alih-alih membatalkan seluruh batch.

---

## Struktur

```
src/kanal/
├── config.py              # setting dari environment; polling policy ada di sini
├── cli.py                 # kanal ingest | load | status
├── ingest/
│   ├── sources.py         # feed registry + peta taksonomi, lengkap dengan alasannya
│   ├── normalize.py       # canonicalisation URL, article_key, pembersihan teks
│   ├── fetch.py           # conditional GET, backoff, circuit breaker
│   ├── parse.py           # bytes feed → Article, tahan terhadap kegagalan
│   ├── land.py            # tulis append-only berpartisi, idempotent
│   └── run.py             # satu siklus, dan laporan jujur tentangnya
└── warehouse/
    ├── duck.py            # setting DuckDB yang dipin; writer/reader terpisah
    ├── schema.py          # DDL raw_articles + _load_log
    └── loader.py          # landing zone → DuckDB, idempotent & incremental

dbt/
├── seeds/taxonomy_map.csv     # channel → kanal, dengan confidence + alasan
├── models/staging/            # stg_articles: cast, derive, mark — tidak filter
├── models/intermediate/       # cluster duplikat + join taksonomi
├── models/marts/              # fct_articles, mart_source_health
├── tests/                     # 4 singular test: kontrak datanya
└── macros/generic_tests.sql   # accepted_range, not_empty_string, unique_combination

scripts/make_fixture.py        # landing zone sintetis untuk CI
```

Landing zone: `data/raw/source={source}/dt={YYYY-MM-DD}/{feed}-{HHMMSS}.jsonl`

Dipartisi per sumber dan per tanggal UTC supaya satu hari bisa diproses ulang
secara terpisah, dan supaya DuckDB serta pyarrow bisa membaca pohonnya langsung
dengan *partition pruning*. Satu file per feed per siklus menjaga penulisan
tetap *append-only* — tidak ada yang pernah ditulis ulang, sehingga run yang
terputus tidak bisa merusak apa yang sudah mendarat.

---

## Konfigurasi

Semua setting dibaca dari environment dengan prefiks `KANAL_`; nilai default di
bawah ini yang dipakai oleh scheduled job.

| Variable | Default | Fungsi |
|---|---|---|
| `KANAL_DATA_DIR` | `./data` | Root landing zone dan warehouse |
| `KANAL_REQUEST_TIMEOUT_S` | `20` | Timeout per request |
| `KANAL_INTER_REQUEST_DELAY_S` | `1.0` | Jeda antar feed |
| `KANAL_MAX_RETRIES` | `3` | Percobaan per feed saat gagal sementara |
| `KANAL_BREAKER_FAILURE_THRESHOLD` | `3` | Kegagalan sebelum feed dilewati |
| `KANAL_BREAKER_COOLDOWN_S` | `3600` | Lama feed dilewati setelah breaker aktif |
| `KANAL_DUCKDB_MEMORY_LIMIT` | `1GB` | Batas memori DuckDB |
| `KANAL_DUCKDB_THREADS` | `2` | Thread DuckDB |

---

## Testing

**66 test, semuanya offline.** Respons feed di-*mock*; suite yang bergantung
pada penerbit sedang hidup adalah suite yang gagal karena alasan di luar
kodenya.

Bobotnya diarahkan ke hal-hal yang mudah salah tanpa ketahuan:

- canonicalisation URL — semua varian nyata menyusut ke satu key, dan input yang
  tidak layak **ditolak** alih-alih dikembalikan dalam bentuk rusak yang akan
  menyebabkan tabrakan key
- idempotency landing zone, termasuk setelah restart proses dan setelah baris
  terakhir terpotong karena proses dimatikan di tengah penulisan
- idempotency dan sifat incremental loader
- pembersihan HTML dan boilerplate, termasuk markup yang rusak
- query umur artikel per sumber, yang mengunci temuan evergreen di atas

CI menjalankan ruff, ruff format, mypy `--strict`, dan pytest pada tiap push.

---

## Belum dibangun

Disebutkan di sini supaya cakupannya terbaca jelas, dan supaya README ini tidak
bisa disalahartikan sebagai deskripsi sistem yang sudah jadi:

orkestrasi (Dagster) · publikasi dataset ke Hugging Face · *near-duplicate
clustering* dengan MinHash · empat kandidat model · *evaluation harness* ·
*promotion gate* · serving · *confidence cascade* · deteksi drift · dashboard.

Satu hal yang **diketahui belum beres**: CNN tidak menghasilkan apa pun saat
ingestion berjalan dari runner GitHub, padahal normal dari mesin lokal. Siklus
berikutnya akan melaporkan status HTTP-nya, dan itu yang menentukan responsnya.
IP datacenter Amerika diblokir sementara IP residensial Indonesia tidak akan
menjadi temuan nyata tentang pasokan datanya, bukan bug yang bisa ditambal.

## Lisensi

MIT — lihat [LICENSE](LICENSE). Teks artikel tetap milik penerbitnya
masing-masing; repository ini hanya menyimpan metadata hasil sindikasi dan
menautkan balik ke sumber aslinya. Lihat [`DATA.md`](DATA.md).

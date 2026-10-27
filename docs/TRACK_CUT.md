# track_quality v2によるtrack cut

実装は `trackcut.py`、一括処理は `shell/trackcut.sh`。

```bash
~/master/shell/trackcut.sh INPUT_DIR OUTPUT_DIR --max-r 1 --invalid-r-policy keep
python -B ~/master/trackcut.py -i input.h5 -o output.h5 --max-r 1 --invalid-r-policy keep --dry-run
```

保持条件は有効な `r < max_r`（等号は除去）。`keep/drop/error` はstate欠損・r算出不能の扱い。
`--require-valid-covariance` を付けた場合だけ、共分散全体の `covariance_valid=True` も要求する。
χ²、hit数、再構成頂点role、500 mm truth未所属による暗黙の除外は行わない。

`--maximumTime 14 --minimumPt 0.3` を明示すれば、既存timingcutと同じtime/pT条件を併用できる。
既存の `master/timingcut.py` は追加groupを落とすため、この処理の前後へそのまま連結しない。
フォルダ一括処理は `bash shell/trackcut.sh INPUT_DIR OUTPUT_DIR --max-r 1 --invalid-r-policy keep`。
再帰処理ではなく、指定ディレクトリ直下の `.h5` を対象とする。

作業ディレクトリに依存せず以下を実行できる。

```bash
~/master/shell/trackcut.sh \
  INPUT_DIR OUTPUT_DIR --max-r 1 --invalid-r-policy keep
```

全ての既知groupとgroup属性を保持する。denseな所属表は同じ行maskを適用し、
疎なqualityとPFOリンクは残存入力へ再対応する。target表、PFO energy、event truth、
sourceのinput件数は元の値を保持し、出力を `training_only=True` とする。
`event_eval.h5_index` のみ出力event順へ再採番する。
groupのバッファ形式は再packするが、残存値・dtype・追加schema属性を維持する。
未知group、品質v1、参照不整合、既存出力への暗黙上書きを拒否する。

初回の比較候補は `max-r=1, invalid-r-policy=keep`。
その後に `0.3/0.15` や算出不能除去を別々に比較する。最適閾値やJER改善は未検証。
v2実H5は当地で未確認。以下の検証は添付仕様に沿った合成H5を対象とする。

## cut比較の提案

| 条件 | 意味 |
| --- | --- |
| 元H5 | cutなしの比較基準。 |
| `--max-r 1 --invalid-r-policy keep` | rを計算できるtrackについて曲率相対誤差100%以上を除去。算出不能の影響を分離する初回条件。 |
| `--max-r 0.3 --invalid-r-policy keep` | より厳しい比較条件。 |
| `--max-r 0.15 --invalid-r-policy keep` | Pandora公開コードの既定値に近い比較条件。ただしここではcalo stateを評価しており、Pandoraの選別を再現する条件ではない。 |
| `--max-r 1 --invalid-r-policy drop` | 算出不能も除く影響を別に調べる条件。 |

`--invalid-r-policy error` は算出不能があれば出力を作らず停止する入力点検用。
`--require-valid-covariance` の有無も独立に比較し、rの条件へ暗黙に混ぜない。
同じイベント・truth・readoutでreco JER、平均応答、track保持率、粒子/energy回収率を評価する。
特に高p・forward・V0娘・displacedを分ける。正しいtrackを失う可能性があるため、
χ²だけ、flat hit数、再構成親role、500 mm未所属を最初の除去条件には含めない。

## 検証

2026-10-26、Awkward 2.1.1 / h5py 3.8.0 / NumPy 1.22.3で19件の統合検証に成功。
15group保持、dense/sparse参照の再対応、空イベント・全track除去、連続cut、欠損policy、
共分散条件、破損参照拒否、入力・既存出力保護、失敗時cleanup、CLI、一括処理を確認。
確認専用ソース・fixture・出力は削除済み。実v2 H5と物理性能は未検証。
本体配置後に `bash -n shell/trackcut.sh` と `python -B trackcut.py --help` も正常終了。

# masterの説明書

コードの操作方法・仕様をまとめています。
文書内の相対パスによるコマンド例は、リポジトリのルート（`~/master`）から実行してください。

```bash
cd ~/master
```

| 説明書 | 内容 |
| --- | --- |
| [TRACK_CUT.md](TRACK_CUT.md) | track_quality v2を用いたH5のtrack cut、CLI・一括処理、比較するcut条件。 |
| [EXTENDED_H5_TRAINING.md](EXTENDED_H5_TRAINING.md) | extended H5の学習入力、one-hot、入力選別とtruth未対応行の扱い。 |
| [PARTICLE_HEADS.md](PARTICLE_HEADS.md) | 粒子種別のenergy headとPIDの設定・利用方法。 |
| [CLUSTER_ENERGY_POOLING.md](CLUSTER_ENERGY_POOLING.md) | cluster集約型energy回帰の構造・教師・学習設定。 |
| [PANDORA_EVAL_ANALYSIS.md](PANDORA_EVAL_ANALYSIS.md) | Pandora/GNN評価、ROOT出力と比較の条件。 |

## 今後の文書配置

- コードの操作方法・仕様は原則としてこの`docs/`に置き、索引を更新する。
- 同じ機能の変更は対応する既存説明書へ反映し、日付や作業ごとの説明書を増やさない。
- 文書内のリンクは文書の位置を基準に、コマンドの実行パスはリポジトリルートを基準に記載する。
- ディレクトリ固有のREADMEは、そのディレクトリの入口として現在の位置を維持する。
- 研究上の調査・比較・結論は`~/research/studies/`のテーマ別記録へ保存する。
- 文書を追加・移動したら、この索引と関連リンク、コード内の案内文字列を更新する。

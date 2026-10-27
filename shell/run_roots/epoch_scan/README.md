# モデル・全epoch ROOT/JER評価

通常は各serverへ自分で入り、`setting.sh`でモデル・energy・epoch・GPUを選んで実行します。
共有`/data`へ保存します。SSH接続やジョブスケジューラへの登録は行いません。

## setting.shで担当範囲を手動実行する

`setting.sh`先頭の`model / energy / epochs / gpus / workers_per_gpu`を設定します。
現在の初期値は`5head_pid_onehot / 200 GeV / epoch 0〜19 / GPU 0,1 / 各GPU 10 writer`です。
`worker_mode`の既定は`persistent`で、同じworkerを複数epochの評価に使います。

```bash
cd /data/murata/master/shell/run_roots/epoch_scan
bash setting.sh
```

全保存epochを対象にする場合:

```bash
bash setting.sh --epochs all --energy 200 --gpus 0 --workers-per-gpu 20
```

`epochs=0:20`は0〜19です。`all`は計画作成時点で保存済みの全checkpointを固定し、再開時には追加しません。
`-1`はepoch指定に使用できません。
設定に従ってenergy専用の計画を作成し、このserverで常駐workerを起動します。
通常実行はforegroundです。長時間動かす場合は使用中のterminal管理方法や`nohup`等を使用してください。
JER集計は`--collect`による別処理で、ROOT作成中には実行しません。

`workers_per_gpu`は1 GPUで同時に動かすwriterの数です。GPU `0,1`と`10`なら合計最大20プロセスが動きます。
以前の`xargs -P10`と同様に複数writerを並行実行し、CPU処理待ちの間も他のwriterがGPUを使えます。
新しい計画では1ジョブが1 epoch・1方式（recoまたはtruth）の全選択H5を処理し、1 ROOTへ直接保存します。
空いたworkerへ次のジョブを割り当てるため、複数epochも同時に進みます。
実際の並列数は未完了ジョブ数までです。GPUメモリやCPU使用量に合わせて`--workers-per-gpu`で調整できます。

### epoch間でworkerを再利用する

通常の`bash setting.sh --epochs all`が常駐方式を使用します。各workerはPython・Torch・ROOTの読み込みとCUDA環境の初期化を最初の1回だけ行い、担当ジョブが終わると次のcheckpointへ進みます。
モデル構造と入力・head等の設定が互換ならモデル本体を再利用し、各checkpointの重みを`strict=True`で読み込みます。互換でない設定へ切り替わる場合はモデルを作り直します。
重みの読み込みと推論は各checkpointについて実行します。各workerは互いに独立しており、`workers_per_gpu`件まで同時に処理します。
完了済みジョブは従来と同じ判定でskipし、処理が必要なジョブがあるworkerだけwriter子プロセスを起動します。

既存の`save_root.py`はそのまま残してあり、直接実行する使い方も維持します。
常駐workerも既存writerの計算・詳細ROOT出力を使います。新しいscanでは複数H5を1 ROOTへ直接保存する専用入口を使用します。
入力H5はジョブごとに読み込みます。入力のepoch間cacheやJER専用軽量出力は使用しません。
変更前のplanは、引き続き入力H5ごとにROOTを保存する従来の入口を使います。

「1ジョブごとにwriterを起動する」方式を使う場合:

```bash
bash setting.sh --epochs all --worker-mode subprocess
```

`worker_mode=${WORKER_MODE:-persistent}`の変更や`WORKER_MODE=subprocess bash setting.sh`でも選択できます。
worker方式は評価条件hashや出力先を変更しないため、同じplan・完了済みROOTを使って切り替えられます。
実行中の旧プロセスは自動で切り替わりません。Ctrl-Cで停止して終了を待ち、通常のコマンドで再開してください。

| energy | 自動files-per-flavor | 固定uds入力のevent数の目安 |
| --- | ---: | ---: |
| 40 / 91 GeV | 3 | 4,500 |
| 200 / 350 GeV | 8 | 4,800 |
| 500 GeV | 17 | 5,100 |

各flavorのH5が必要数に満たない場合はエラーです。通常recoとtruth clusteringの両方のROOTを作ります。

共有`setting.sh`を書き換えずに担当を指定することもできます。

```bash
MODEL=5head_pid_onehot ENERGY=200 EPOCHS=all GPUS=0 WORKERS_PER_GPU=20 bash setting.sh
```

異なるserverで異なるenergyやモデルを担当する場合も、それぞれのserverで設定して起動します。
同じモデル・run・energyに対して異なるepoch範囲やファイル数の評価計画を作る場合は、後述の`--tag`等で保存先を分けます。

### 保存先

既定のdirectory構造は次のとおりです。

```text
/data/suehara/mldata/pfa/murata/output/jer_epoch_scan/
  <model>_<run_id>/
    200GeV/
      reco/
        epoch000.root
        epoch001.root
      truth/
        epoch000.root
        epoch001.root
      metrics.csv
      _scan/
        plan.json
        filelists/inputs.txt
        logs/local/<job_id>.log
        state/<job_id>.json
        summaries/epoch000/<method>/<readout>/
    500GeV/
      ...
```

モデル・run・energyはそれぞれ1回だけpathに入ります。`manual`・`default`・`epochs_all_fpf..._<hash>`は付けません。
1 epoch・1方式につき1 ROOTです。各ROOTに全flavor・全選択H5を含めます。
`truth/`はtruth clusteringの結果です。通常の採用判断には`reco/`を使用します。
`_scan/plan.json`にはepoch集合、files-per-flavor、入力一覧、checkpoint、推論条件、設定hashを保存します。
`_scan/filelists/`はwriterへ渡す入力一覧、`_scan/logs/`は実行ログ、`_scan/state/`は完了判定と重複実行防止の管理情報です。
`_scan/summaries/`とenergy直下の`metrics.csv`は`--collect`の実行後に作成します。

`SCAN_OUTPUT_BASE=/data/...`または`--output-base /data/...`で保存先の基点を変更できます。
`--scan-dir /data/...`を指定すると、そのdirectoryを直接使用します。
`--tag NAME`は明示的に別の評価を保存する場合だけ使用し、基点とモデルdirectoryの間に`NAME/`を追加します。既定のtagはありません。

### 確認・再開・集計

```bash
bash setting.sh --epochs all --dry-run --max-jobs 2  # 計画を作成/再利用し、実行コマンドだけ表示
bash setting.sh --epochs all --plan-only            # 計画作成だけ
bash setting.sh --epochs all --status               # 保存済みの状態を確認。未作成ならエラー
bash setting.sh --epochs all --collect              # 完成したROOTのJER集計
```

各操作では、計画を作成したときと同じモデル・energy・epoch指定・ファイル数等を使用します。
`--files-per-flavor 0`で自動件数を上書きして全H5を使用できます。

同じ設定で再実行すると、同じdirectoryの計画を再開します。
GPU番号や`workers_per_gpu`、`worker_mode`は変更できます。完了済みジョブを再作成しません。
並列数等を変更する前に、動いているforeground実行をCtrl-Cで停止して終了を待ってください。

同じ保存先に異なる評価条件を指定すると、上書きや別hash directoryの作成はせずエラーにします。
例えばepoch `0:20`の計画があるenergyに`--epochs all`を指定すると条件不一致になります。
別条件や追加学習後のcheckpoint一覧を評価する場合は、`--tag next`・別の`--output-base`・別の`--scan-dir`で保存先を分けてください。
元configの他モデルだけを変更しても、選択中のモデルの再開には影響しません。

### 変更前に作成したscan

energy直下に旧`plan.json`がある場合は、自動的に従来方式で再開します。
実行中の旧scanを新形式へ自動変換しません。稼働中の200 GeVもそのまま継続できます。
旧planも`--scan-dir`で指定すれば常駐方式で再開できます。

```bash
old_scan='/data/suehara/mldata/pfa/murata/output/jer_epoch_scan/<model>_<run_id>/200GeV'
bash setting.sh --epochs all --energy 200 --scan-dir "$old_scan"
bash setting.sh --epochs all --energy 200 --scan-dir "$old_scan" --collect
```

`old_scan`には実際のdirectoryを指定してください。モデル・epoch等も旧planと同じ条件にします。
古い`manual/default/.../epochs_..._<hash>/`に残るplanは、そのdirectoryを明示してください。
同じenergy directoryに旧planがあれば、`--scan-dir`を省略しても旧planを再利用します。
完成済みの分割ROOTを新形式にする場合は、別途結合処理が必要です。ROOTの単純なrenameでは変換できません。

完成済み91 GeVを変換する実行例です。GPU推論は行いません。

```bash
scan='/data/suehara/mldata/pfa/murata/output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602/91GeV'
python3 -B migrate_epoch_files.py --scan-dir "$scan" --dry-run
python3 -B migrate_epoch_files.py --scan-dir "$scan" --workers 2
```

`--dry-run`は全jobの完了と旧ROOT一覧を確認し、変換予定のファイル数を表示します。
本実行は各epoch・方式のROOTを結合し、元のイベント・truth energy・各energy readoutが一致することを確認します。
全出力の検証後に新plan・完了記録・`metrics.csv`を公開し、元の分割ROOTを片付けます。
結合中は元ROOTと新ROOTが共存するため、新ROOT相当の空き容量を使用します。
中断した場合は同じ移行コマンドで再開します。進捗は`_scan/migration.json`に保存します。
移行中のenergyは`setting.sh`から実行せず、移行完了後に通常の`--status`や再開コマンドを使用します。
未完了・実行中のscanは移行できません。上記は実行方法の例であり、91 GeVの移行完了を示すものではありません。

`--worker-mode`、`--workers-per-gpu`、`--files-per-job`、`--pid-threads`、`--pipeline-depth`、`--root-chunk-events`も指定できます。
`--files-per-job`は旧planの分割条件です。新形式では全選択H5を1ジョブにまとめるため、この値で分割しません。
`PYTHON_BIN=/path/to/python`で使用するPythonを指定します。
`run_scan.sh --model ... --energy ... --epochs ... --gpus ...`による直接指定も可能です。
直接`run_scan.sh`、`run_persistent.py`、`run_parallel.py`を起動する場合の`--workers-per-gpu`既定値は1で、`setting.sh`が既定で10を指定します。

## ファイル

| ファイル | 役割 |
| --- | --- |
| `setting.sh` | GPU・energy・model・epoch・並列数の編集用入口 |
| `run_scan.sh` / `run_setting.py` | 自動FPF選択、単一serverでの計画作成・再開・実行 |
| `scan_layout.py` | 手動scanのモデル・energy別directory構造を設定 |
| `epoch_jobs.py` | 新形式の出力・実行を既存runnerへ接続し、旧planの動作を維持 |
| `epoch_writer.py` / `epoch_root.py` | 全選択H5の直接保存と、既存ROOTのepoch単位結合・検証 |
| `epoch_collect.py` | epoch単位ROOTの入力対応を検証し、JERを集計 |
| `migrate_epoch_files.py` | 完成済みの分割ROOTを再推論せずに結合・比較し、新形式へ移行 |
| `collect_scan.py` | 手動scanの構造に合わせて既存JER集計処理を呼び出す |
| `scan_config.json` | 対象runとモデル固有の入力・readout設定 |
| `plan_scan.py` | 従来の汎用planner。checkpoint・入力・server担当を固定 |
| `run_persistent.py` | 各GPUで複数workerを常駐させ、epoch間で再利用する既定runner |
| `persistent_worker.py` | 同じPython/CUDA環境と互換モデルを使って既存writer処理を実行 |
| `run_parallel.py` | 従来方式。ジョブごとにwriterを起動して並列実行 |
| `run_server.py` | 既存計画と共通処理を維持する従来runner（1 subprocess/GPU） |
| `collect_jer.py` | 分割ROOTを読み、既存定義のJERとCSVを作成する共通処理 |
| `scan_common.py` | ハッシュ、状態ファイル、出力確認の共通処理 |

既存の学習コード、writer、`run_roots`の他directoryは変更しません。

## 従来の汎用plannerを直接使う場合

以下は複数モデル・energyを1つの計画に含め、server名ごとに担当を分ける従来の方法です。
`setting.sh`による上記の手動実行とは別の入口です。
`plan_scan.py`を直接使う計画は、従来の`roots/<model>/<run>/epoch_NNN/<energy>GeV/<method>/shard_NNN/`構造を維持します。
集計は従来どおり`collect_jer.py`を使用します。
`モデル/run × epoch`の全energy・reco/truthは同じserverへ割り当てます。

## 1. 評価計画を一度だけ作る

各serverで同じPython環境（Torch・ROOT（PyROOT）・NumPy・Awkward・h5py等を使える環境）を有効にします。
計画の作成だけはPython標準ライブラリで動きます。

```bash
cd /data/murata/master
scan=/data/suehara/mldata/pfa/murata/output/jer_epoch_scan/scan_001

python3 -B shell/run_roots/epoch_scan/plan_scan.py \
  --out "$scan" \
  --servers serverA:1,serverB:1,serverC:2 \
  --files-per-flavor 5 \
  --files-per-job 20
```

- `serverA`等は担当名です。実際のhostnameと一致する必要はありません。
- `:2`は`:1`のserverに対する処理能力の目安です。担当量を多く割り当てます。GPU数が倍なら、まず2を目安にできます。
- `--epochs`の既定は`all`で、**設定した各runで保存済みの全epoch**を含めます。以後追加されるcheckpointはこの計画には入りません。
- `--files-per-flavor 5`は各energyの`uu/dd/ss`から名前順に5 H5ずつ使います。全H5を使う場合は`0`、または引数を省略します。H5ごとのevent数が違うため、同じファイル数でもenergy別のevent数は異なります。
- `--files-per-job`は一度のモデルロードで処理するH5数です。event数ではありません。
- 計画は`plan.json`と共有入力リストとして保存されます。入力path・size・mtime、checkpoint SHA256、コードSHA256を固定します。
- 既存の`--out`は上書きしません。入力・設定・評価対象を変えるときは新しいscan名を使います。

`scan_config.json`には、既存launcherに対応する次の5系列を登録しています。

| 名前 | 学習run |
| --- | --- |
| `2head` | `2026_09_29_120252`（clustering head BN eps=`1e-5`） |
| `5head_pid` | `2026_10_04_162529` |
| `5head_pid_onehot` | `2026_10_09_154602` |
| `5head_pid_onehot_truth_gather` | `2026_10_19_154730` |
| `5head_pid_onehot_preicted_gather` | `2026_10_19_183348` |

既定は40/91/200/350/500 GeV、通常recoとtruth clustering、tbeta=0.9、td=0.5です。
`truth_gather`等は学習時の所属を表すモデル名で、評価時の`reco/truth_clustering`とは別です。
`preicted`は既存directory名の綴りに合わせています。

対象を限定する例:

```bash
python3 -B shell/run_roots/epoch_scan/plan_scan.py \
  --out /data/suehara/mldata/pfa/murata/output/jer_epoch_scan/scan_subset \
  --servers serverA,serverB \
  --models 5head_pid_onehot,5head_pid_onehot_preicted_gather \
  --epochs 0,5,10:21:2 \
  --files-per-flavor 2
```

epoch範囲はPythonと同じ`開始:終了:間隔`で、終了を含みません。上例は0,5,10,12,...,20です。
指定したepochが対象runにない場合はエラーにします。runごとに存在する全epochを使う通常運用は`--epochs all`です。

他のrunはconfigの`models`に追加し、`name`を一意にしてください。`checkpoint_dir`は絶対path、または`repo`からの相対pathです。
2head/5head、PID、one-hot、座標次元、過去版のBN epsなど、実際のcheckpointに合う設定を使います。
任意のconfigは`--config /data/.../my_scan.json`で指定できます。
`nstart`は各H5の開始event、`nend`は各H5で読む**件数**（`-1`は末尾まで）です。全体のevent上限ではありません。
入力rootをモデル別に変更する設定も可能ですが、異なるevent集合のモデルは同じJER比較に混ぜず、別scanで扱ってください。

## 2. 各serverで担当を実行する

まず実行するコマンドを確認します。dry-runは推論も状態ファイルの作成もしません。

```bash
cd /data/murata/master
scan=/data/suehara/mldata/pfa/murata/output/jer_epoch_scan/scan_001

python3 -B shell/run_roots/epoch_scan/run_parallel.py \
  --plan "$scan/plan.json" --server serverA --gpus 0,1 \
  --workers-per-gpu 10 --dry-run --max-jobs 2
```

server Aで:

```bash
nohup python3 -B shell/run_roots/epoch_scan/run_parallel.py \
  --plan "$scan/plan.json" --server serverA --gpus 0,1 \
  --workers-per-gpu 10 --pid-threads 2 > "$scan/serverA.console.log" 2>&1 &
```

server Bでは`--server serverB`、server Cでは`--server serverC`に変更します。
`--gpus`はそのserverで使用するGPU番号です。指定値を子プロセスの`CUDA_VISIBLE_DEVICES`に渡し、内部deviceは`cuda:0`になります。
UUID指定も可能です。`--python /path/to/python`または`PYTHON_BIN`でwriterのPythonを指定できます。
`--gpus cpu`はCPU動作確認用です。

各GPUは`--workers-per-gpu`件まで同時処理し、空いたworkerがそのserverの次のジョブを取ります。
上例はGPU 0,1でそれぞれ最大10ジョブ、合計最大20ジョブです。
CPU thread数は`--pid-threads`、prefetchは`--pipeline-depth`、ROOT書込単位は`--root-chunk-events`で調整できます。
同じ共有planのジョブを二重投入した場合はファイルロックで拒否します。共有filesystemはLinuxの`flock`をサポートする必要があります。

出力:

```text
scan_001/
  plan.json
  filelists/
  roots/<model>/<run_id>/epoch_019/91GeV/reco/shard_000/<入力名>.root
  roots/<model>/<run_id>/epoch_019/91GeV/truth_clustering/shard_000/<入力名>.root
  logs/<server>/<job_id>.log
  state/<job_id>.json
```

## 3. 状態確認・再実行

```bash
python3 -B shell/run_roots/epoch_scan/run_parallel.py --plan "$scan/plan.json" --status
```

通常は同じ実行コマンドを再実行します。完了状態と出力のsize/mtimeが一致するジョブを省略し、失敗・未完了・変更された出力を再処理します。
`--recheck`を付けると、完了ジョブも既存writerのROOT構造・metadata検証を通します。対応する完成ROOTはwriter側で再利用します。
明示的にROOTを作り直すときだけ`--replace`を指定します。
失敗ジョブがあると終了codeは非0になります。他ジョブは既定で続行し、`--fail-fast`では新規ジョブの開始を止めます。

```bash
python3 -B shell/run_roots/epoch_scan/run_parallel.py \
  --plan "$scan/plan.json" --server serverA --gpus 0 \
  --job-id JOB_ID
```

停止したserverの担当を別serverで引き継ぐときは、元の担当名（例`--server serverA`）を指定します。
元serverの処理が終了したことを確認して引き継いでください。ジョブのロックは終了時にOSが解放します。残った`.lock`ファイルは削除不要です。
SIGINT/SIGTERMでは新規ジョブを止め、実行中の子writerを終了してからロックを解放します。
評価中にコード・checkpoint・入力を変更すると拒否します。保存済みの版へ戻すか、新しいscanを作ります。

## 4. 全server終了後にJERを集計する

共有pathなので、いずれか1台のCPU環境から実行できます。

```bash
python3 -B shell/run_roots/epoch_scan/collect_jer.py --plan "$scan/plan.json"
```

ROOTは分割のまま読みます。`hadd`でevent番号を混ぜたり、分割ごとのJERを平均したりしません。
既存`macro/src/pid_jet_resolution_summary.h`を直接使い、各モデル・epoch・energy・reco/truthのevent energy分布を合わせて計算します。
指標は既存figuresと同じ`100*sqrt(2)*RMS90(Ereco_event)/Mean90(Ereco_event)`、最小RMS90%区間です。
中央・forward・角度bin、Mean90、RMS90、event数、overflow、valid情報を保存します。
この指標は固定energy sample用で、nnqqの広いtruth energy分布へそのまま適用するものではありません。

- 全モデルに`saved`を保存します。2headは既存のtrack/calo readoutを使います。
- PIDモデルにはさらに`raw_head`と`track_replaced`を保存し、既存figuresと比較できます。
- 出力は`scan_001/metrics.csv`と、`summaries/<model>/<run>/epoch_XXX/<energy>GeV/<method>/<readout>/`以下のROOT・txtです。
- groupごとの`coverage.json`に入力・event集合のhashと欠損を記録します。異なるモデル・epochの完全なevent集合が一致することも確認します。
- ROOT欠損・破損・設定違いは既定でエラーです。`--allow-incomplete`は調査用の部分集計で、該当groupの`eligible_for_ranking`をfalseにします。
- 集計途中の状況を確認する場合も、最終表とは別のCSVを指定してください。

```bash
python3 -B shell/run_roots/epoch_scan/collect_jer.py \
  --plan "$scan/plan.json" \
  --models 5head_pid_onehot --epochs 14 \
  --output-csv "$scan/metrics_onehot_epoch14.csv"
```

checkpoint選択用の固定評価集合と、最終性能を確認する未使用testは分けて管理します。
既に繰り返し比較した固定uds sampleは、未使用testとして扱わないでください。


## ROOT入出力のPyROOT方針

Pythonで新規実装・修正するROOTファイル入出力にはPyROOTを使用します。メタデータ取得、TTree件数・イベント番号の検証、ROOT結合、JER集計も対象です。

`epoch_root.py`のROOT生成・出力検証・結合に加え、`collect_jer.py`、`epoch_collect.py`、`macro/save_pid_energy_readouts.py`の読み書きもPyROOTで行います。解析側の共通I/Oは`macro/pyroot_io.py`を使用します。通常の`setting.sh`によるROOT生成も`--collect`も、ROOT入出力のためのuprootは不要です。NumPy・Awkward・h5pyは既存の数値計算・H5入力用に使用します。元の`save_root.py`の入口と出力計算は維持します。

PyROOTへのI/O互換変更を既存scanへ反映する場合は、対象コードの旧hashを確認し、planの対応する`implementation`／`execution_implementation`だけを更新します。評価入力・checkpoint・physics設定・job ID・完成済みROOT・metricsは維持し、hash検査は省略しません。対応するplanには旧／新hashと理由の監査記録を残します。実行中のscanには適用せず、全対象scanの完了とprepare／GPU／job lockの取得を確認してから適用します。

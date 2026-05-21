# eval-jmteb-v2

JMTEB v2 の Retrieval タスク群で text embedding モデル (dense / sparse) を評価するための実行ハーネスです。設定を JSON ファイル 1 枚で与え、`eval.py` を実行するだけで全タスクのベンチマークが回ります。

## このプロジェクトの目的

JMTEB v2 (公式の評価フレームワーク) は dense な embedding モデルの評価を前提としており、SPLADE のような sparse (疎ベクトル) モデルをそのままでは評価できません。本リポジトリは、JMTEB v2 の Retrieval タスク資産をそのまま使いつつ、sparse モデルも同じ土俵で評価できるようにするためのハーネスです。

具体的には、`SparseEncoder` (sentence-transformers) を mteb の `EncoderProtocol` に適合させる薄いラッパー (`SpladeMTEBWrapper`) を挟むことで、JMTEB v2 の `JMTEBV2Evaluator` に sparse モデルを渡せるようにしています。これにより SPLADE 系の sparse モデルと従来の dense モデルを、同一のタスク・指標・出力形式で比較できます。dense モデルの評価にもそのまま使えます。

## 特長

- dense モデル (`mteb.get_model`) と sparse モデル (SPLADE / `SparseEncoder`) を `is_sparse` フラグ 1 つで切り替え
- 評価対象のタスク・バッチサイズ・系列長をタスク別に JSON で指定
- レジューム対応 (summary と details が揃ったタスクは自動スキップ)
- タスク単位の try/except により、1 タスクの失敗 (OOM 等) で全体が止まらない
- JMTEB 本体の clone と依存パッケージの install は実行時に自動で行われる

## 必要環境

- Python 3.10 以降
- CUDA 対応 GPU (sparse モデルの大規模 corpus 評価には十分な VRAM を推奨)
- git (JMTEB を実行時に clone するため)
- Hugging Face にアクセスできること (非公開モデルを使う場合は `HF_TOKEN` を設定)

依存パッケージ (`mteb==2.4.2` 固定, sentence-transformers, transformers 等) と JMTEB 本体の clone は、`modules/env_setup.py` の `setup_environment()` が実行時に自動で行います。手動の pip install は不要です。

> 注意: `mteb` は 2.4.2 に固定しています。mteb v2.4 の encode inputs は `List[Dict[str, List[str]]]` (column-batch) 形式で、`SpladeMTEBWrapper.encode` がこの形を前提に展開しています。バージョンを上げると形式が変わり動作しなくなる可能性があります。

## 使い方

```bash
# eval.py と同じディレクトリの config.json を使う
python eval.py

# 任意の config を明示指定 (第1引数)
python eval.py configs/config.template.json
```

全タスクの評価は合計で長時間 (構成によっては 9〜10 時間規模) かかります。バックグラウンドで実行し、ログの `[evaluator]` 行と `Encoding Batch N/M` を監視すると進捗を追えます。途中で中断しても、完了済みタスクは次回スキップされて再開されます。

## config スキーマ

`configs/config.template.json` が雛形です (sparse モデル, 全 11 タスク)。dense モデルの例は `configs/config.dense_example.json` を参照してください。

| キー | 型 | 説明 |
| --- | --- | --- |
| `target_models` | list[str] | 評価対象モデル ID のリスト。複数指定すると順に評価する (旧 `target_model` 単数も後方互換で受ける)。 |
| `is_sparse` | bool | true なら SPLADE 等を `SpladeMTEBWrapper` でロード、false なら `mteb.get_model` でロードする。 |
| `target_tasks` | list[str] | 評価する JMTEB Retrieval タスク名のリスト。 |
| `task_max_seq_length` | dict[str, int\|null] | タスク別の最大系列長。null のタスクは override せず、モデルの素の最大コンテキスト長で評価する。実効値は `min(指定値, モデルの最大コンテキスト長)`。 |
| `task_batch_size` | dict[str, int] | タスク別のバッチサイズ。唯一のバッチ指定元で、`target_tasks` の全タスクに非 null の値が必須。 |
| `output_root_dir` | str | 結果出力ルート。実際の出力先は `output_root_dir/<モデル ID の "/" を "__" に置換した名前>/`。 |

グローバルな `batch_size` / `max_seq_length` は廃止しています。バッチサイズは必ず `task_batch_size` でタスク別に指定してください。

### 対応タスク

JMTEB v2 の Retrieval タスク群を対象としています。

```
NLPJournalTitleAbsRetrieval.V2
NLPJournalTitleIntroRetrieval.V2
NLPJournalAbsIntroRetrieval.V2
NLPJournalAbsArticleRetrieval.V2
MintakaRetrieval
JaGovFaqsRetrieval
JaqketRetrieval
JaCWIRRetrieval
MultiLongDocRetrieval
MIRACLRetrieval
MrTidyRetrieval
```

## 出力

モデルごとに `output_root_dir/<モデル ID の "/" を "__" に置換した名前>/` 配下へ出力されます。

- `<TASK>.json` — JMTEB 形式の summary (main_score など)
- `details/<TASK>.json` — mteb 形式の詳細スコア
- `model_meta/<TASK>.json` — モデルメタ情報

JMTEB の split 不一致 (例: MIRACL は `scores.dev` のみ) で summary が空 (`{}`) になることがありますが、実行終了時に details の `scores.<split>[0].main_score` から自動復元します。

## アーキテクチャ

`eval.py` (薄いエントリポイント) が config を読み、`modules/` の公開 API を順に呼びます。

1. `env_setup.setup_environment()` — JMTEB を `<cwd>/JMTEB` に clone/pull し `src` を sys.path へ追加、固定版パッケージを install、PyTorch アロケータ env (`expandable_segments:True`) を設定、mteb / sentence_transformers のロガーを INFO に上げる。この呼び出し後でないと mteb / jmteb は import できないため、`model_loader` / `evaluator` 内では import を関数内に遅延させている。
2. `model_loader.build_model(model_id, is_sparse)` — sparse なら `SpladeMTEBWrapper`、dense なら `mteb.get_model()`。
3. `evaluator.run_evaluation()` — タスクを順に評価。per-task のバッチサイズと系列長を適用しつつ、レジューム判定・GPU メモリ解放・summary 復元を行う。

### モジュール構成

```
eval.py                     エントリポイント (config 読み込み → modules 呼び出し)
modules/
  __init__.py               公開 API の re-export
  env_setup.py              JMTEB clone / pip install / 環境変数設定
  model_loader.py           dense / sparse のモデルロード、max_seq_length の取得・設定
  sparse_wrapper.py         SparseEncoder を mteb の EncoderProtocol に適合させるラッパー
  evaluator.py              タスク単位の評価ループ (レジューム・OOM 耐性・summary 復元)
  tasks.py                  タスク名 → JMTEB タスクオブジェクトへの解決
  jsonio.py                 JSON 読み書きユーティリティ
configs/
  config.template.json      sparse モデルの雛形 (全 11 タスク)
  config.dense_example.json dense モデルの例
```

## OOM が出たとき

OOM が出たタスクの `task_batch_size` の該当エントリだけを下げて (例: 32 → 24 → 16 → 12 → 8) 再実行してください。他タスクを巻き添えにしないよう、OOM したタスクのエントリだけを下げます。完了済みタスクは自動スキップされるため、途中から再開されます。

なお sparse モデルで corpus が大きく NNZ も大きいタスク (例: JaCWIR) では、cuSPARSE の SpGEMM が `insufficient resources` で落ちることがあります。`SpladeMTEBWrapper.similarity` は corpus を dense 展開して sparse×dense matmul へ自動でリトライします。

# g38クラスタ用ジョブスクリプト

g38クラスタでFIONA-WORKLOADを実行するためのSlurmジョブスクリプト。

## 前提条件

- g38にSSHログイン済み
- `module load slurm/20.02.5` 実行済み（または~/.bashrcに設定済み）

## 使い方

### Spikeシミュレーション

```bash
# デフォルト（benchmark_transformer）
sbatch run_spike.sh

# アプリケーション指定
sbatch run_spike.sh benchmark_transformer
sbatch run_spike.sh sentiment_bert
```

### ジョブ状態確認

```bash
squeue --me          # 自分のジョブ一覧
cat fiona-<JOB_ID>.out  # 標準出力確認
cat fiona-<JOB_ID>.err  # エラー出力確認
```

### ジョブキャンセル

```bash
scancel <JOB_ID>
```

## 出力ファイル

| ファイル | 内容 |
|---------|------|
| `fiona-<JOB_ID>.out` | 標準出力 |
| `fiona-<JOB_ID>.err` | エラー出力 |

## カスタマイズ

スクリプト内の`#SBATCH`オプションを変更可能：

```bash
#SBATCH -t 1:00:00        # 制限時間（1時間）
#SBATCH -w g38            # 特定ノード指定
#SBATCH --mail-type=END   # 終了時メール
#SBATCH --mail-user=xxx@example.com
```

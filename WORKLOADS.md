# FIONA-Workload ワークロード一覧

このドキュメントでは、FIONA-Workloadリポジトリに含まれるワークロードとライブラリについて説明します。

## 目次

1. [概要](#概要)
2. [ワークロード一覧](#ワークロード一覧)
   - [基本テスト](#基本テスト)
   - [数学演算テスト](#数学演算テスト)
   - [ニューラルネットワークテスト](#ニューラルネットワークテスト)
   - [機械学習アプリケーション](#機械学習アプリケーション)
3. [ライブラリ](#ライブラリ)
4. [実行方法](#実行方法)

---

## 概要

FIONA-Workloadは、フォトニックバックエンドを使用したニューラルネットワークアプリケーションとカーネルを提供します。RISC-V GNUツールチェーンでコンパイルされ、FIONA-Spikesim（ISAシミュレータ）またはFIONA-V（RTLシミュレーション/FPGA）上で実行できます。

```
fiona-workload/
├── app/           # ワークロードアプリケーション
├── lib/           # 共有ライブラリ
├── build/         # コンパイル済みELFファイル
└── Makefile       # ビルドスクリプト
```

---

## ワークロード一覧

### 基本テスト

| ワークロード | ディレクトリ | 説明 |
|-------------|-------------|------|
| **hello_fiona** | `app/hello_fiona/` | 最小限の動作確認用サンプル |
| **test_relu** | `app/test_relu/` | ReLU命令の動作確認テスト |

#### hello_fiona

環境セットアップの確認用。単純に"hello fiona!"を出力します。

```c
int main(int argc, char** argv) {
    printf("hello fiona!\n");
    return 0;
}
```

#### test_relu

FIONAのReLU命令（`RELU_V`）の動作確認。正負の値を含む入力に対してReLU関数を適用し、期待値と比較します。

---

### 数学演算テスト

| ワークロード | ディレクトリ | テスト対象 |
|-------------|-------------|-----------|
| **math_ealu** | `app/math_ealu/` | ベクトル加算・減算（Element-wise ALU） |
| **math_palu** | `app/math_palu/` | 内積、MVM、行列積（Photonic ALU） |
| **math_nlu** | `app/math_nlu/` | ReLU活性化関数（Nonlinear Unit） |
| **math_misc** | `app/math_misc/` | max, argmax等の補助演算 |

#### math_ealu (Element-wise ALU)

行列とベクトルの要素ごとの加算・減算をテスト。

```c
// テスト関数
test_tiled_matrix_vector_add();  // mat + vec (各行に加算)
test_tiled_matrix_vector_sub();  // mat - vec (各行から減算)
```

#### math_palu (Photonic ALU)

フォトニックコアで実行される主要演算をテスト。**これらの演算がフォトニックハードウェアで加速されます。**

```c
// テスト関数
test_mat_transpose();           // 行列転置
test_tiled_dotprod();           // 内積（ベクトル×ベクトル）
test_tiled_mvm();               // 行列ベクトル積（MVM）
test_tiled_matmul_transpose();  // 行列積
```

#### math_nlu (Nonlinear Unit)

非線形活性化関数のテスト。

```c
// テスト関数
test_tiled_matrix_relu();  // ReLU: max(0, x)
```

#### math_misc

その他の補助演算のテスト。

```c
// テスト関数
test_tiled_matrix_vector_max();  // 各行の最大値
test_matrix_vector_argmax();     // 各行の最大値インデックス
```

---

### ニューラルネットワークテスト

| ワークロード | ディレクトリ | テスト対象 |
|-------------|-------------|-----------|
| **nn_linear** | `app/nn_linear/` | 全結合層（Linear/Dense） |
| **nn_conv** | `app/nn_conv/` | 2D畳み込み層（Conv2d） |
| **nn_pool** | `app/nn_pool/` | 最大プーリング層（MaxPool2d） |
| **nn_pad** | `app/nn_pad/` | パディング処理（Padding2d） |

#### nn_linear

全結合層（Linear Layer）のテスト。Irisデータセットの一部を使用。

- **入力**: 30サンプル × 4特徴量
- **出力**: 30サンプル × 10ニューロン
- **演算**: `Y = X @ W^T`（MVMベース）

#### nn_conv

2D畳み込み層のテスト。

- **入力**: 1バッチ × 3チャンネル × 7×7
- **カーネル**: 2出力チャンネル × 3入力チャンネル × 4×4
- **パラメータ**: stride=3, padding=0

#### nn_pool

最大プーリング層のテスト。2D/4Dテンソルの両方をテスト。

- **カーネルサイズ**: 2×2
- **パディング**: 1

#### nn_pad

ゼロパディング処理のテスト。2D/4Dテンソルの両方をテスト。

---

### 機械学習アプリケーション

| ワークロード | ディレクトリ | 説明 |
|-------------|-------------|------|
| **mlp_iris** | `app/mlp_iris/` | MLP推論（Irisデータセット、デバッグ出力付き） |
| **mlp_iris_infer** | `app/mlp_iris_infer/` | MLP推論（フォトニックモデル選択対応） |
| **mlp_large** | `app/mlp_large/` | 大規模MLP推論（性能ベンチマーク用） |

#### mlp_iris / mlp_iris_infer

**Multi-Layer Perceptron (MLP)** による Iris データセット分類。

```
アーキテクチャ: 4 -> 10 -> 3
- 入力層: 4ニューロン（sepal/petal length/width）
- 隠れ層: 10ニューロン + ReLU
- 出力層: 3ニューロン（3クラス分類）
```

**mlp_iris_infer** は環境変数 `FIONA_PHOTONIC_MODEL` でフォトニックモデルを選択可能:
- `ideal`: 理想的な数値計算
- `noisy`: ガウスノイズ追加
- `quantized`: 量子化ノイズ
- `mzi_nonlinear`: MZI非線形性
- `mzi_realistic`: 現実的なMZIモデル（位相誤差、熱クロストーク等）
- `all_effects`: すべてのノイズを適用

#### mlp_large

**大規模 Multi-Layer Perceptron (MLP)** による性能ベンチマーク。

```
アーキテクチャ: 128 -> 256 -> 128 -> 64 -> 10
- 入力層:   128ニューロン
- 隠れ層1: 256ニューロン + ReLU
- 隠れ層2: 128ニューロン + ReLU
- 隠れ層3:  64ニューロン + ReLU
- 出力層:   10ニューロン（10クラス分類）

パラメータ数: 74,368
- FC1: 256 × 128 = 32,768
- FC2: 128 × 256 = 32,768
- FC3:  64 × 128 =  8,192
- FC4:  10 ×  64 =    640

バッチサイズ: 64
総MAC演算数: 4,759,552
```

**特徴:**
- 決定論的疑似乱数で生成された重み（再現性確保）
- 5ビット量子化重み、6ビット入力
- 層ごとの統計情報（min, max, mean）出力
- 詳細な性能サマリー

**mlp_large** も環境変数 `FIONA_PHOTONIC_MODEL` でフォトニックモデルを選択可能です。

**実行例:**
```bash
# 理想モデル
spike --extension=fiona pk ../fiona-workload/build/mlp_large.elf

# ノイズモデル
FIONA_PHOTONIC_MODEL=noisy spike --extension=fiona pk ../fiona-workload/build/mlp_large.elf
```

---

## ライブラリ

### lib/base/

基本的なヘッダーとFIONA命令定義。

| ファイル | 説明 |
|---------|------|
| `config.h` | 設定定義（`elem_t = int16_t`等） |
| `instr.h` | FIONA RoCC命令マクロ |
| `rocc.h` | RoCCインターフェース定義 |

#### 主要な命令マクロ (instr.h)

```c
// ベクトル算術演算
ADD_V(vd, v1, v2)    // vd = v1 + v2
SUB_V(vd, v1, v2)    // vd = v1 - v2
MUL_VS(vd, r1, v2)   // vd = r1 * v2 (スカラー×ベクトル)
DIV_VS(vd, r1, v2)   // vd = r1 / v2

// 活性化関数
RELU_V(vd, v1)       // vd = max(0, v1)
TANH_V(vd, v1)       // vd = tanh(v1)
SIGMOID_V(vd, v1)    // vd = sigmoid(v1)

// メモリ操作
LOAD_V(vregnum, src)   // ベクトルレジスタにロード
STORE_V(vregnum, dst)  // ベクトルレジスタからストア

// 設定
SET_VLEN(r1)         // ベクトル長設定
SET_MAT(r1)          // 行列レジスタ設定

// フォトニック演算（高速化対象）
DOTP(rd, v1, v2)     // 内積
MVM(vd, v1)          // 行列ベクトル積
```

### lib/math/

フォトニックバックエンドの数学演算ライブラリ。

| ファイル | 説明 |
|---------|------|
| `ealu.h` | 要素ごとの加減算 |
| `palu.h` | 内積、MVM、行列積（フォトニック加速） |
| `nlu.h` | 非線形関数（ReLU等） |
| `misc.h` | max, argmax等 |
| `contrib.h` | 寄与演算 |
| `all.h` | 全数学演算のインクルード |

### lib/nn/

ニューラルネットワークモジュールライブラリ。

| ファイル | 説明 |
|---------|------|
| `common.h/.cc` | 共通ユーティリティ |
| `linear.h/.cc` | 全結合層 |
| `conv.h/.cc` | 畳み込み層 |
| `pool.h/.cc` | プーリング層 |
| `norm.h/.cc` | 正規化層 |
| `backprop.h` | バックプロパゲーション |
| `weights_io.h/.cc` | 重み入出力 |
| `mlp_iris_weights.h` | 学習済みMLP重み |

### lib/utils/

ユーティリティ関数。

| ファイル | 説明 |
|---------|------|
| `pprint.h` | 行列・ベクトルの表示関数 |

---

## 実行方法

### ビルド

```bash
# 環境変数設定
export RISCV="/path/to/riscv-tools"

# ビルド
cd fiona-workload/
make

# 出力: build/*.elf
```

### 実行（Spikeシミュレータ）

```bash
# 環境設定
source /path/to/fiona_undergraduate/setup-env.sh

# fiona-spikesimディレクトリに移動
cd /path/to/fiona-spikesim/

# 実行
spike --extension=fiona pk ../fiona-workload/build/<workload>.elf
```

**実行例:**

```bash
# Hello World
spike --extension=fiona pk ../fiona-workload/build/hello_fiona.elf

# MLP推論（デフォルトモデル）
spike --extension=fiona pk ../fiona-workload/build/mlp_iris_infer.elf

# MLP推論（ノイズモデル指定）
FIONA_PHOTONIC_MODEL=noisy spike --extension=fiona pk ../fiona-workload/build/mlp_iris_infer.elf

# MLP推論（現実的MZIモデル）
FIONA_PHOTONIC_MODEL=mzi_realistic spike --extension=fiona pk ../fiona-workload/build/mlp_iris_infer.elf
```

### 実行（Verilator RTLシミュレーション）

```bash
cd chipyard/sims/verilator/
make run-binary-debug \
    CONFIG=FionaRocketConfig \
    BINARY=../../../fiona-workload/build/<workload>.elf \
    LOADMEM=1 -j
```

---

## ワークロード選択ガイド

| 目的 | 推奨ワークロード |
|------|-----------------|
| 環境確認 | `hello_fiona` |
| 命令テスト | `test_relu`, `math_*` |
| NN層テスト | `nn_linear`, `nn_conv`, `nn_pool` |
| 推論デモ | `mlp_iris`, `mlp_iris_infer` |
| フォトニックモデル比較 | `mlp_iris_infer` + `FIONA_PHOTONIC_MODEL` |
| 性能ベンチマーク | `mlp_large` |
| 大規模推論テスト | `mlp_large` + `FIONA_PHOTONIC_MODEL` |

---

## 参考資料

- [FIONA論文](https://iccad.com/) - ICCAD 2023
- [FIONA-Spikesim](https://github.com/hkust-fiona/fiona-spikesim) - ISAシミュレータ
- [FIONA-V](https://github.com/hkust-fiona/fiona-v) - RTL実装
- [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)

---

**最終更新**: 2025-12-18

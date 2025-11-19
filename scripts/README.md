# MERIT 实验脚本使用指南

## 核心脚本（9个）

### 1. `multi_seed_run.py`
多随机种子实验运行器，自动统计 Mean±Std。

### 2. `find_best_params.sh`
超参数搜索（3×3×3=27 个配置）

### 3. `run_all_datasets.sh`
一键运行 3 个数据集（APAVA, PTB, PTB-XL）的完整实验

### 4. `run_baselines.sh`（可选）
统一运行 MedGNN / iTransformer / FEDformer / ECGFM / ECGFounder / FORMED（按数据集自动裁剪，支持 APAVA / PTB / PTB-XL），并生成日志与 CSV。

### 5. `run_ablation.sh`
5 个变体消融实验

### 6. `summarize_all_datasets.py`
结果汇总（含 LaTeX 输出）

### 7. `evaluate_uncertainty.py`
不确定性评估（ECE、Selective Prediction 等）

### 8. `analyze_uncertainty.py`
不确定性分析（噪声鲁棒性、分布、拒绝实验、案例）

### 9. `README.md` + `QUICK_GUIDE.md`
使用文档

---

## 实验流程（4 步）

### Step 0: 超参数搜索

```bash
bash MERIT/scripts/find_best_params.sh APAVA 0
bash MERIT/scripts/find_best_params.sh ADFD-Sample 0
bash MERIT/scripts/find_best_params.sh PTB 0
bash MERIT/scripts/find_best_params.sh PTB-XL 0

# 查看最佳配置
cat results/param_search/APAVA/best_config.txt
cat results/param_search/ADFD-Sample/best_config.txt
cat results/param_search/PTB/best_config.txt
cat results/param_search/PTB-XL/best_config.txt
```

搜索空间：
- 学习率: 1e-4, 1.5e-4, 2e-4
- Lambda_view: 0.5, 1.0, 1.5
- Lambda_pseudo: 0.2, 0.3, 0.5

---

## 然后三步完成主实验

### Step 1: 用最佳配置更新 `run_all_datasets.sh`
根据 `best_config.txt`，修改 `run_all_datasets.sh` 中各数据集的参数。

### Step 2: 运行完整实验

```bash
bash MERIT/scripts/run_all_datasets.sh
```

输出：3 个数据集 × 10 seeds × 5 指标

### Step 3: Baseline 对比

使用各项目官方的多数据集脚本：

- iTransformer（支持 `--with_swa`）：
  ```bash
  python -m iTransformer.scripts.run_ecg_subject_all \
    --datasets APAVA,PTB,PTB-XL \
    --apava_root dataset/APAVA \
    --ptb_root dataset/PTB \
    --ptbxl_root dataset/PTB-XL \
    --gpu 0 --seeds 3 \
    --out_dir results/iTransformer_final_all_datasets \
    --with_swa
  ```
- MedGNN：
  ```bash
  python -m MedGNN.scripts.run_subject_all \
    --datasets APAVA,PTB,PTB-XL \
    --apava_root dataset/APAVA \
    --ptb_root dataset/PTB \
    --ptbxl_root dataset/PTB-XL \
    --gpu 0 --num_seeds 3 \
    --out_dir results/MedGNN_final_all_datasets
  ```

> 依赖提示
> - MedGNN / iTransformer：需要 `MedGNN/MedGNN` 仓库及依赖。
>   - iTransformer 的 ECG 分类脚本在 `iTransformer/scripts/run_ecg_subject_all.py`（支持 `--with_swa`）。
>   - MedGNN 的 ECG 多数据集脚本在 `MedGNN/scripts/run_subject_all.py`。
> - FEDformer：需要 `FEDformer/FEDformer` 仓库。
> - ECGFM：默认读取 `ECGFM/checkpoint/last_11597276.ckpt`（如存在）或官方 `ecg_fm/mimic_iv_ecg_physionet_pretrained.pt`。
> - FORMED：若 TimesFM 只有 `model.safetensors`，脚本会自动转换为 `.pth` 并缓存。

---

## 数据集与默认配置（参考）

| Dataset | lr | epochs | 其他关键参数 |
|---------|-----|--------|--------------|
| APAVA | 1e-4 | 150 | λ=(λ_fuse=1.0, λ_view=1.0, λ_pseudo_loss=0.3), anneal=50 |
| ADFD | 1e-4 | 150 | e_layers=6, dropout=0.2 |
| PTB | 1.5e-4 | 120 | wd=0, λ_pseudo_loss=0.2, anneal=40 |
| PTB-XL | 1.5e-4 | 120 | wd=0, λ_pseudo=0.3, anneal=40 |

注：最终以脚本内参数为准，可按需调整。

---

## 消融实验（快速版）

- 一键多数据集（APAVA, PTB, PTB-XL），较少轮次：
```bash
python -m MERIT.scripts.run_ablation_all \
  --datasets APAVA,PTB,PTB-XL \
  --root_paths APAVA=dataset/APAVA,PTB=dataset/PTB/PTB,PTB-XL=dataset/PTB-XL/PTB-XL \
  --gpu 0 \
  --seeds 41,42 \
  --max_epochs 80 \
  --patience 10
```
- 变体：Full / w/o Evidential Fusion (`--agg mean --no_pseudo`) / w/o Pseudo-view (`--no_pseudo --lambda_pseudo_loss 0.0`) / w/o Frequency (`--no_freq`) / w/o Difference (`--no_diff`)
- 输出：`results/ablation_all_quick/<DATASET>/<variant>.csv`

---

## 不确定性实验（导出与分析）

- 训练并导出（EviMR 与 Softmax）见 `QUICK_GUIDE.md` 示例。
- 仅评估导出（复用已训练 checkpoint）：
```bash
python -m MERIT.scripts.multi_seed_run \
  --root_path dataset/APAVA \
  --data APAVA \
  --gpu 0 \
  --train_epochs 150 --patience 20 \
  --e_layers 4 --dropout 0.1 --nodedim 10 \
  --resolution_list 2,4,6,8 \
  --seeds "41,42,43" \
  --eval_only
```
- 评估与对比：`evaluate_uncertainty.py` 与 `compare_selective.py`

---

## 参考与说明

- 外部依赖仓库请根据其官方 README 安装配置（MedGNN、iTransformer、FEDformer、ECGFM、FORMED 等）。
- 本仓库脚本默认以 GPU 单卡运行，数据集根目录与超参数可按项目需求调整。


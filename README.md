# 2026 全国大学生数学建模竞赛 · B 题

题目：无线电干扰源的快速自动定位与清除。

## 目录

- `B题.pdf`、`附件1.docx`、`附件2.docx`：题目及官方接口说明。
- `Jammers-simulator-win64.7z`：提供的官方模拟器压缩包。
- `question1/`：测向交集、定位区域直径、可复现仿真和测试。

## 快速开始

建议使用 Python 3.10 或更新版本。在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m question1.simulate --seed 42 --points 4
```

生成的图片、CSV 和 JSON 位于 `question1/output/`，默认不纳入 Git。详细数学说明、函数接口和误差模型见 [第一问说明](question1/README.md)。

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest question1/tests -q
```

本仓库的第一问仿真独立运行，不启动官方模拟器、不消耗正式测试次数。

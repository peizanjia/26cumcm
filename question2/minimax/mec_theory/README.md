# 最小覆盖圆：命题、证明与配图

`mec_rigorous_section.tex` 可替换原论文“5.3 最小覆盖圆指标及其求法”小节。
证明包含：覆盖顶点与覆盖凸包等价；最小覆盖圆存在唯一；圆心位于边界顶点凸包的充要条件；至多三个支撑点；完整枚举正确性；直径圆提前返回判定。

## 插入论文

将本目录的三个文件 `mec_preamble.tex`、`mec_rigorous_section.tex`、`mec_support_diagram.tex` 放在论文主 tex 同目录。

导言区加入（只加一次）：

```latex
\input{mec_preamble.tex}
```

将原来的 5.3 小节替换为：

```latex
\input{mec_rigorous_section.tex}
```

使用 XeLaTeX 编译两遍。主论文已有命题环境时，可复用已有环境并替换 `mecproposition` / `meccorollary`，避免重复定义。
图采用 TikZ，插入正文时无需额外 PNG 或 PDF。

## 源码交付

本目录交付可嵌入论文的证明、宏包和 TikZ 源码。将三份 tex 文件放在论文主文件同目录，按上述说明插入并用 XeLaTeX 编译。

## 实现口径

现有 mechanistic.py 已实现完整两点、三点枚举与全顶点覆盖检查。
“先检查全局直径圆”在此作为可增加的优化步骤说明，不声称现有代码已经实现该提前返回。
几何定理针对精确运算，不把浮点容差和未来角度采样误写为严格上界保证。

# 非晶聚合物本构辨识代码

本仓库配套论文 Origins of non-uniqueness in the identification of
internal-variable constitutive models for glassy polymers，提供本构更新、
加载控制、完整历史灵敏度、有限标定和残留状态比较的程序。

作者为 Zhiyang Ni、Tianxiang Lan、Yaodong Jiang、Peidong Wu、Yueguang Wei。
通讯作者仍为 Tianxiang Lan，邮箱 lantianxiang111@163.com。

建议使用 Python 3.11，按英文 README 安装依赖。

```bash
./reproduce.sh list
./reproduce.sh smoke
./reproduce.sh verify
```

smoke 和 verify 不需要实验数据，用于检查安装、本构更新、控制方程、
参数缩放等价性和完整状态重启。其余分析的入口及前提见 docs/WORKFLOWS.md。

共享范围是代码、数据读取接口、数值检验和加载方案；完整论文比较还需要
外部文献数字化曲线及部分归档计算文件。本仓库未包含这些输入，因此不能
把代码仓库说成包含全部原始数据或全部结果。输入结构见 docs/EXTERNAL_DATA.md。

输出保存在本地且由 Git 忽略。原数值归档及其校验记录保留不动。

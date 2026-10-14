# Aircraft 仿真器与源码简版

本包为 `Aircraft_simple.zip`，对应 2026-10-10 打包时的项目版本 `b3f82e9a06a7c0a9af5be3e8c67dfbd1cefd7298`。主项目 Python 源码和原实验入口与最新版项目一致。

这是源码包，包含仿真器、算法适配器、第三方算法代码、测试、实验入口和安装说明。实验结果、运行日志、论文、图片、预训练模型、虚拟环境和 `old` 不随包提供；若要核对论文原始数据，请使用另一份完整复现包。

## 1. 项目做什么

研究多飞行器在局部观测和通信不完整条件下的协同决策。每架飞机由自身智能体选择高层动作；训练阶段可使用集中式价值信息，执行阶段的分散策略读取本机观测。MAT 是集中式联合观测对比方法，其信息条件不同。

仿真器提供决策级多机环境，以及 JSBSim F-16 六自由度验证后端。高层动作包括保持、支援、覆盖、恢复、退出和关键任务，连续控制分量包括转向、爬升和加减速。在线流程如下：

```text
局部观测 -> 信念更新与 STL 规则风险估计
         -> 规则—智能体依赖图与离散冲突仲裁
         -> 连续残差 QP/SQP 安全修正
         -> 环境执行、奖励与离线真值评测
```

规则覆盖物理安全、信息授权、故障资源、团队协同、任务连续性和效率偏好。系统通过规则优先级与最小必要动作修正协调冲突；不能理解为已经完成真实飞控认证或所有场景下的安全保证。

## 2. 安装

建议使用 Windows 64 位、Python **3.12.9** 和 Git for Windows。解压到短路径，如 `D:\AircraftSimple`，进入解压得到的 `Aircraft` 目录，再运行：

```powershell
python --version
git --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-reproduce.txt
& .\.venv\Scripts\python.exe verify_package.py
& .\.venv\Scripts\python.exe -m pytest -q
```

依赖需要联网安装。计算使用 CPU，不要求 CUDA。依赖文件固定了包验证使用的直接库版本。其他 Python/库版本未逐一验证。

官方方法适配器会检查第三方 commit，所以必须保留每个 `external_methods/inbox/*/SOURCE/.git` 隐藏目录，并安装 Git。第三方版本信息随包提供，无需重新克隆官方仓库。省略的是第三方展示图片、无关原始 benchmark 资产和预训练权重；第三方浅克隆对象库用于验证原始版本，仍予保留。

## 3. 快速启动

检查从训练到执行的完整小规模流程：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_package_smoke.ps1 -PythonExecutable .\.venv\Scripts\python.exe
```

该脚本依次运行测试、本文方法及六种官方方法的小规模训练、JSBSim checkpoint 验证和逐步案例记录。运行后会新建 `results/package_smoke`，其中的数据是本次运行生成的，不是随包的论文结果。重复运行请指定不同的 `-RunRoot`，例如 `-RunRoot results/package_smoke_2`。

如只想运行本文方法和仿真器：

```powershell
& .\.venv\Scripts\python.exe run_trainable_baselines.py --methods proposed_belief_stl_conflict_qp --seeds 11 --updates 2 --episodes 4 --horizon 10 --warmstart-epochs 5 --output results/proposed_demo
& .\.venv\Scripts\python.exe run_high_fidelity_validation.py --backend jsbsim --seeds 11 --episodes 2 --horizon 10 --updates 2 --decision-horizon 10 --output results/jsbsim_demo
```

`surrogate` 可用于低成本接口检查，但不能代替 JSBSim 六自由度实验。

## 4. 正式实验入口

从源码重新训练主实验、两类消融和 JSBSim 验证：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_protocol.ps1 -PythonExecutable .\.venv\Scripts\python.exe -RunRoot results/reproduction_main
```

再加上风险校准、压力测试、恢复与介入时机实验、案例日志及动态优先级机制验证：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduce_all.ps1 -PythonExecutable .\.venv\Scripts\python.exe -RunRoot results/reproduction_full
```

完整实验比小规模启动检查耗时长。源码包没有现成 checkpoint，`run_checkpoint_conflict_traces.py`、`run_risk_calibration_audit.py`、`run_high_fidelity_checkpoint_comparison.py` 需要先训练模型并指定对应模型目录；上述完整脚本已经按依赖顺序编排。

随机种子和协议一致不保证跨机器、库版本和线程配置得到逐位相同的训练结果或运行时间。高冲突实验区分初始安全保持与违规后恢复；动态优先级定向实验用于观察动作仲裁差异，不能单独推导总体安全净收益。

## 5. 源码目录

```text
aircraft_sim/       仿真环境、奖励、规则、信念、STL、依赖图、QP、训练与评测
external_adapters/  MAPPO、HAPPO、HATRPO、MACPO、MAPPO-Lagrangian、MAT 等桥接
external_methods/   固定版本第三方算法源码及原许可证/来源说明
tests/              回归测试
run_*.py            训练、仿真、校准、压力测试和案例记录入口
analyze_*.py        统计分析入口
run_final_protocol.ps1  主实验批量运行入口
reproduce_all.ps1       主实验及补充实验编排
```

核心文件包括 `aircraft_sim/env.py`、`rules.py`、`belief.py`、`rule_graph.py`、`qp_shield.py`、`mappo.py`、`evaluation.py`、`scenarios.py` 和 `jsbsim_env.py`。

GCBF+ 是另一个物理安全专项，源码和适配接口予以保留。它不属于七方法完整规则主表；其可选推理需要独立 JAX 环境与预训练文件，简版包不提供预训练文件。相关权重可从完整复现包或固定版本上游仓库取得，安装步骤见对应 `REPO_INFO.md`。

`package_manifest.json` 保存源码版本、文件清单和 SHA-256；`verify_package.py` 校验解压后的文件及第三方版本。简版不会恢复原项目中已删除的文件。

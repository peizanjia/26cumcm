# 第四问演练和正式测试入口

主脚本是 `question4/main.py`，实现位于 `question4/official_main.py`。它直接调用当前冻结的 `tuned_joint.Planner`，默认加载 `question4/tuned_joint/best_config.json` 中选定031的完整45参数。不会加载训练第一名027，也不会使用类默认参数替代缺失字段。此前独立100场481.40秒/源是本地合成结果，不是官方成绩。

## 启动方式

在项目根目录运行，使用现有 `.venv`，无需激活虚拟环境：

```powershell
# 演练：未提供 robot_id 时会提示输入当前登录模拟器的参赛队号
& .\.venv\Scripts\python.exe -X utf8 .\question4\main.py

# 正式：先在模拟器选择“问题4正式测试”
& .\.venv\Scripts\python.exe -X utf8 .\question4\main.py --test-type formal --formal-confirmed
```

也可以使用 `-m question4.main`。给出 `--robot-id YOUR_LOGIN_ID` 可省略输入提示，或设置环境变量 `JAMMER_ROBOT_ID`。第三问已有的 `--once`、`--base-url`、`--params`、`--output`、`--wait-enter-s`、`--enter-poll-s`、`--request-timeout`、`--retries`、`--quiet` 均保留相同用途。默认地址 `http://127.0.0.1:2026`；默认每秒轮询进入，网络单次超时15秒，同一请求最多尝试3次。

运行流程：

1. 在模拟器登录，选择问题4对应的演练/正式模块；脚本模式参数只标记日志并检查正式确认，不会切换模拟器的题号或模块。
2. 点击一次“开始测试”。倒计时/接口尚未开放时，脚本继续等待；收到 `/enter` 的 `accepted=true` 后执行完整算法。
3. 原点扫描全部20频道，然后按冻结算法动态补盲、定位和清除；仅在实际停点测量，每步根据公开反馈重规划。
4. 完成后发送 `/exit`，保存本局摘要和公开历史，输出本局检测到的目标数，然后等待下一次手动开始。加 `--once` 则运行一局后退出。
5. Ctrl+C 停止脚本；活动局仍可通信且未超时的时候尝试 `/exit`，已完成的请求/响应和运动历史始终逐步落盘。异常局也保存已知进度，不能冒称完整完成。

可以加 `--launch-simulator --simulator-exe "实际路径\jammers-simulator.exe"` 先打开模拟器；仍需手动登录和开始。脚本退出后保留模拟器窗口，供查看结果和完成其自身日志上传。

## 填表所需的目标个数

**只在一局结束时输出一次计数，不逐个目标实时打印。** 例如：

```text
本局檢測到的干擾源個數：13
```

同一频道最多一个源，因此将所有真实 `direction`、`near` 反馈或成功清除对应的频道去重；清除后的目标仍保留在累计发现数中。每局目录另保存 `detected_count.txt`，内容只有数字和换行，便于复制到论文表格。

`summary.json` 的 `detected` 是这个数字，`detected_channels` 是对应频道，`cleared` 是实际成功清除数，`remaining_detected` 是发现但尚未清除数。若 `complete=true`，算法已依据公开观测形成完整结束证书，此时 `inferred_total_sources=detected` 可用作本局源总数。中途退出、超时或异常时，`inferred_total_sources=null`，检测数只是已发现数，不能保证等于场景总数。

附件2的四个接口均不提供源的全向/定向类型。`svd_deg` 是测点指向源的带误差示向度，不是源的发射朝向，不能直接据此统计真实类型。因此 `omnidirectional_count`、`directional_count` 为 `null`，`type_counts_available=false`，不填造假的0或猜测值。附件1说明演练结束后的模拟器界面可以显示全向/定向真值；正式测试的界面和接口都不提供这两类真实数量。本脚本只使用公开HTTP反馈。

## 每局保存内容

默认目录为 `question4/official_runs/practice/run_时间_序号/` 或 `question4/official_runs/formal/run_时间_序号/`，不同局互不覆盖：

| 文件 | 内容 |
|---|---|
| `detected_count.txt` | 仅本局累计检测到的目标数 |
| `summary.json` | 完成状态、检测/清除数、总虚拟耗时、每个已清除源的平均耗时、失败原因 |
| `run_configuration.json` | 完整45参数、参数哈希、模式、服务地址、robot_id |
| `http_requests.jsonl` | 逐次发送请求与原始响应；同一动作重试保留同一ID，并记录连接错误 |
| `commands.jsonl` | 实际运动起终点、频道、反馈、费用和选择原因 |
| `frames.jsonl` | 每个动作后的公开位置范围、状态及路线提示，可供后续回放 |
| `decisions.jsonl` | 候选点评分、沿途频道价值、实际选择理由 |
| `map.json` | 最终公开估计地图和逐频道未知域 |

`http_requests.jsonl` 的 `event=request` 先于发送落盘；`event=response` 保存收到的完整反馈。断线重试时会有多个 request 事件，但同ID动作只计一次。`accepted=false` 返回的时间0不会覆盖当前虚拟时刻，也不会更新位置/信念。记录中的无穷诊断值保存为JSON `null`，不改变规划计算。

运行时使用 `/enter` 返回的实际 `remaining_real_duration_s`，发送动作前预留2秒退出余量；计算过程不另外添加虚拟耗时。超时或接口关闭后不通过 `/exit` 查询原因。若某次动作响应最终丢失，摘要只统计确认收到的反馈，原始发送和重试仍可在HTTP日志中检查。

这些自存的公开日志不是模拟器生成的正式加密 `.jlog`；正式日志仍通过模拟器导出。默认运行目录已加入Git忽略列表。

## 本地验证

只在临时本机端口启动合成HTTP服务器，不连接2026或启动官方测试。协议测试覆盖：真实冻结参数加载、格式/模式检查、进入等待、同ID断线重试、拒绝反馈不更新状态、微秒计时、清除不切换测向频道、到时退出、中断保存、结束后统计、连续两局状态重置以及两种入口形式。

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m pytest question4/test_official_main.py -q
```

联调时使用 `--http-context local --base-url http://127.0.0.1:临时端口`，摘要显式标记 `local_synthetic_http`；默认 `official` 才记录演练/正式上下文。题号4和测试类型不是HTTP请求字段，绝不向接口增加未知字段。

2026-09-13最终核验：本入口20项测试及现有101项第四问测试共121项通过（17.31秒）。另完成两个完整本地HTTP案例：20296075清除16/16、187个动作；20296040清除10/10、361个动作，第二例需要依靠逐频道补盲证书判断没有遗漏。两例的动作位置、频道、反馈结果和示向度均与冻结基准逐条相同；累计时间按官方微秒格式输出。第二例最终仅输出一次计数10，`detected_count.txt` 内容为 `10`。这些联调不增加独立样本量、不更新481.40的既有算法统计，也没有连接官方模拟器。

完整HTTP联调可以用测试中的临时服务器复现（需要 `requirements-dev.txt` 中的pytest）：

```powershell
@'
from pathlib import Path
from question4.test_official_main import synthetic_http, args_for
from question4.official_main import run_one, load_parameters, DEFAULT_PARAMS
for index, seed in enumerate((20296075, 20296040), 1):
    with synthetic_http(seed=seed) as server:
        args = args_for(server, Path("question4/official_runs/local_reproduction"))
        result = run_one(args, index, load_parameters(DEFAULT_PARAMS))
        assert result["complete"] and all(s.cleared for s in server.simulator.sources)
        assert result["detected"] == len(server.simulator.sources)
'@ | .\.venv\Scripts\python.exe -X utf8 -
```

上面仅在运行结束后的测试断言访问合成场景真值，正式主程序仍只接收HTTP反馈。

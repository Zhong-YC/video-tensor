# 批量下载示例说明

任务清单使用标准 JSON，不支持 //、# 或块注释。为保留可直接运行的清单，原来的全部成功配置另存为 tasks-all-success.json；注释说明放在本文件中。

以下命令均在项目根目录运行。

全部成功示例的第一项仅填写 id 和 url，省略 mode 和时间，默认完整下载。显式填写 full 也可；时间字段省略或 null 均表示未指定时间。section 任务仍须填写有效的起止时间。

## 中间失败，前后成功：tasks.json

| 顺序 | id | 任务说明 | 预期状态 |
| --- | --- | --- | --- |
| 1 | short-full | 完整下载公开视频 | success |
| 2 | intentional-404 | 故意访问不存在的 MP4 地址，预期网站返回 HTTP 404 | failed |
| 3 | short-section | 下载公开视频的 2–5 秒片段，验证失败后仍继续处理 | success |

```powershell
.\.venv\Scripts\python.exe main.py batch "examples/tasks.json"
$LASTEXITCODE
```

预期汇总：

```text
批量汇总：成功 2，失败 1，未开始 0，取消：否
```

退出码预期为 1，表示本批次包含失败任务；它不表示程序崩溃或第三条没有执行。请紧接运行命令查看 $LASTEXITCODE，避免被其他程序覆盖。

打开控制台给出的 results.json，核对 tasks 数组中的三个 status 依次为 success、failed、success。第二项 error 应包含 404；第一、三项 path 应指向已保存的视频文件。每项结束后汇总都会更新，失败任务的临时文件会清理。

第二条特意使用格式合法的 HTTP URL，在实际访问阶段触发失败。不要把它改为非法 URL、反向时间区间或缺少必填字段：这类配置错误会在整份清单预检查时被拒绝，整批都不会开始下载。

## 原来的全部成功示例：tasks-all-success.json

原配置保留两条任务，没有加入故意失败的地址：

| 顺序 | id | 任务说明 | 预期状态 |
| --- | --- | --- | --- |
| 1 | short-full | 完整下载公开视频 | success |
| 2 | short-section | 下载同一视频的 2–5 秒片段 | success |

```powershell
.\.venv\Scripts\python.exe main.py batch "examples/tasks-all-success.json"
```

预期汇总为成功 2、失败 0、未开始 0、取消：否，退出码为 0。两个清单可直接切换，无需手动注释或删除任务。

以上结果以测试网站正常可访问为前提；故意失败地址若不再返回 404，应检查 results.json 中记录的实际错误。

## 下载完成后批量解码

直接把上述运行输出的 results.json 路径传入，默认将成功文件全部解码，无需逐个复制视频路径：

```powershell
.\.venv\Scripts\python.exe main.py decode-batch "实际输出目录/results.json"
```

两种示例均包含 full 和 section：

- full 视频默认解码整个 10 秒视频，Tensor shape 为 (250, 3, 176, 320)。
- section 文件解码完整的本地 0–3 秒片段，Tensor shape 为 (75, 3, 176, 320)。
- tasks-all-success.json 对应汇总：解码成功 2、失败 0、跳过 0。
- tasks.json 对应汇总：解码成功 2、失败 0、跳过 1（原 404 任务）。

如需只解码 full 视频的 2–5 秒，可选填区间：

```powershell
.\.venv\Scripts\python.exe main.py decode-batch "实际输出目录/results.json" --full-start-time 2 --full-end-time 5
```

此时 full 视频为 75 帧，section 片段仍为 75 帧。两个时间参数必须同时提供或同时省略。Tensor 逐条在内存中生成并输出摘要，不保存 .pt，不修改下载汇总；长视频默认整体解码，需要相应可用内存。

解码前按实际目标区间估计像素内存，默认达到 2 GiB 时询问“是否取消本次解码”：y/yes 跳过当前项，n/no 继续。两份短视频示例均低于默认阈值，不会询问。选择跳过的项单独计入 skipped，后续任务继续，视频文件及下载汇总保留。

无人值守运行时，可明确选择仅跳过达到阈值的项：

```powershell
.\.venv\Scripts\python.exe main.py decode-batch "实际输出目录/results.json" --warn-memory-gib 2 --large-decode-action skip
```

完整参数与 Python API 回调见项目 README 第 4.6、5.5 节。

# video-tensor

**当前版本：MVP v0.21** · **已验证：Windows 11 x64 / Python 3.12.2**

将在线视频或本地视频解码为 PyTorch 视频 Tensor。支持完整下载、在线指定区间下载、本地整段或指定区间解码、串行批量下载，以及读取下载汇总批量解码。

单任务流程：URL → yt-dlp 下载 → TorchCodec 解码 → Tensor。

批量流程：任务清单 → 串行下载 → results.json → 串行解码 → 逐条生成 Tensor。

默认行为：未指定起止时间时，下载完整视频，解码整个目标文件；已下载的区间片段则解码整个片段。单视频下载的模式默认为 `full`，批量下载清单需填写 `"mode": "full"`。主动选择 `section` 区间下载时，仍需成对填写起止时间。

## Quick Start

前提：已安装 Python 3.12 x64、FFmpeg 8.1.2 shared，并将 FFmpeg 的 `bin` 加入 PATH。以下命令在项目根目录的 PowerShell 中运行，每行分别执行。

首次安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

已有可用 `.venv` 时可跳过安装，直接选择下面的流程。

**单个短视频：完整下载后，将全部内容解码为 Tensor。**

```powershell
.\.venv\Scripts\python.exe main.py download "https://www.w3schools.com/html/mov_bbb.mp4" --mode full --decode
```

预期：完整下载 10 秒视频，输出 `shape=(250, 3, 176, 320)`、`dtype=torch.uint8`；Tensor 时间戳为 0–9.96 秒。需要仅下载片段时，使用第 4.1 节的 `section` 命令。

**多个视频：批量下载，再统一解码已有下载结果。**

```powershell
.\.venv\Scripts\python.exe main.py batch "examples/tasks-all-success.json"
```

复制控制台最后显示的 `results.json` 完整路径，填入下一条命令：

```powershell
.\.venv\Scripts\python.exe main.py decode-batch "替换为上一步输出的results.json完整路径"
```

示例包含完整视频和 2–5 秒片段：默认分别解码 10 秒 / 250 帧、3 秒 / 75 帧。两次汇总均应成功 2 项。只需提供一次汇总文件路径，无需逐个复制视频路径。

安装详情见第 1–3 节；全部命令与参数见第 4 节；清单格式和时间规则见第 5 节。

## 项目目录

```text
video-tensor/
├── main.py                 # 统一入口：download / local / batch / decode-batch
├── downloader.py           # 完整下载、在线指定区间下载
├── decoder.py              # 本地视频元信息、帧与区间 Tensor
├── pipeline.py             # 串联单视频下载与解码
├── batch.py                # 下载清单校验、串行执行、下载结果汇总
├── batch_decoder.py        # 读取下载汇总，逐条生成 Tensor
├── examples/
│   ├── tasks-all-success.json # 全部成功的批量下载示例
│   ├── tasks.json             # 成功 → 故意 404 失败 → 成功的容错示例
│   └── README.md              # 两种示例的预期结果
├── tests/                  # 本地解码、下载、批量处理测试
├── requirements.txt        # 固定的 Python 依赖
├── README.md
└── .gitignore
```

## 1. Python 版本

已验证环境为 Windows 11 x64、Python 3.12.2。使用项目独立的 `.venv`；当前为 CPU 解码，无需 CUDA。

| 组件 | 已验证版本 |
| --- | --- |
| Python | 3.12.2 x64 |
| PyTorch | 2.10.0+cpu |
| TorchCodec | 0.10.0 |
| yt-dlp | 2026.8.19 |
| NumPy | 2.5.3 |
| FFmpeg | 8.1.2 shared |

Python 依赖固定在 `requirements.txt`。安装时使用这一组版本；其他操作系统尚未验证。

## 2. FFmpeg 安装要求

安装 **FFmpeg 8.1.2 Windows x64 shared 构建**，例如 `ffmpeg-8.1.2-full_build-shared.7z`。可从 [FFmpeg 下载页](https://ffmpeg.org/download.html) 进入 Windows 构建站。

将解压目录的 `bin` 加入 Windows 用户环境变量 Path。保留完整 shared 库，包括 `ffmpeg.exe`、`ffprobe.exe`、`avcodec-62.dll`、`avformat-62.dll`、`avutil-60.dll` 等。

重新打开终端，检查：

```powershell
ffmpeg -version
ffprobe -version
```

两条命令均应输出版本信息；PyCharm 内置终端可能需要重启 IDE 才能使用新的 PATH。

TorchCodec 需要 FFmpeg 动态库；在线区间下载还会运行 FFmpeg，使用 libx264 编码器。上述 shared 构建包含该编码器。仅安装同名 Python 包不能替代 FFmpeg 程序及动态库。

## 3. pip 安装命令

在项目根目录确认 `python --version` 为 Python 3.12，然后依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

已有可用环境无需重新创建。`requirements.txt` 已配置 PyTorch CPU 安装源。

验证依赖：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; from torchcodec.decoders import VideoDecoder; print('PyTorch:', torch.__version__); print('TorchCodec import OK')"
```

预期依次看到 `No broken requirements found.`、`PyTorch: 2.10.0+cpu`、`TorchCodec import OK`。这一步检查依赖及加载；实际下载和解码可使用 Quick Start 验证。

### PyCharm 可选配置

1. 按 Ctrl + Alt + S 打开设置，搜索“Python 解释器”或 Python Interpreter。
2. 当前解释器应指向 `<项目目录>\.venv\Scripts\python.exe`。
3. 若不正确，选择“添加解释器 → 添加本地解释器 → 已有环境”，选择上述文件并应用。不同版本菜单名称可能略有区别。

这只配置 IDE 的运行/调试解释器，不重新安装依赖，也不保证已有终端自动切换。终端可用以下命令检查实际解释器：

```powershell
python -c "import sys; print(sys.executable)"
```

直接调用 `.\.venv\Scripts\python.exe` 无需激活虚拟环境。

## 4. 怎么运行

所有命令在项目根目录运行。下面使用完整的虚拟环境解释器路径；`URL`、`本地视频完整路径`、`results.json完整路径` 是需要替换的输入，包含空格的路径保留双引号。

| 子命令 | 用途 | 主要输出 |
| --- | --- | --- |
| `download` | 单个在线视频完整下载或区间下载，可接着解码 | 视频路径；开启解码时输出 Tensor 摘要 |
| `local` | 已有本地视频默认整段或指定区间解码 | Tensor 摘要 |
| `batch` | 根据 JSON 清单串行下载多个任务 | 视频文件、逐项状态和 `results.json` |
| `decode-batch` | 根据下载汇总串行解码成功项 | 每项 Tensor 摘要和解码计数 |

### 4.1 单视频下载与下载后解码：download

**完整下载视频，只保存视频文件：**

```powershell
.\.venv\Scripts\python.exe main.py download "URL" --mode full
```

**完整下载视频，再将全部内容解码为 Tensor：**

```powershell
.\.venv\Scripts\python.exe main.py download "URL" --mode full --decode
```

**完整下载视频，再解码原视频的 2–5 秒：**

```powershell
.\.venv\Scripts\python.exe main.py download "URL" --mode full --start-time 2 --end-time 5 --decode
```

**只下载原视频的 4800–4830 秒：**

```powershell
.\.venv\Scripts\python.exe main.py download "URL" --mode section --start-time 4800 --end-time 4830
```

**只下载原视频的 4800–4830 秒，再解码整个下载片段：**

```powershell
.\.venv\Scripts\python.exe main.py download "URL" --mode section --start-time 4800 --end-time 4830 --decode
```

**区间下载时限制所选格式高度，并指定输出目录：**

```powershell
.\.venv\Scripts\python.exe main.py download "URL" --mode section --start-time 4800 --end-time 4830 --max-height 360 --output-dir "work/custom-downloads"
```

| 参数 | 是否必填 / 默认值 | 说明 |
| --- | --- | --- |
| `url` | 必填位置参数 | 单个视频的网页 URL 或媒体直链，必须是 HTTP(S) 地址；位置参数直接写值，不加 `--url` |
| `--mode` | 可选，默认 `full` | `full` 先下载完整视频；`section` 直接请求指定时间段 |
| `--start-time` | `section` 必填 | 原视频的开始秒数，包含起点；支持小数，如 `2.5`；完整下载时用于选择后续解码区间 |
| `--end-time` | 与开始时间成对提供 | 原视频的结束秒数，不包含终点；必须大于开始时间 |
| `--decode` | 可选开关，默认关闭 | 下载后生成 Tensor：full 不填时间时解码全部，填写时间时解码指定区间；section 解码整个下载片段。开关无需填写 true/false |
| `--max-height` | 可选，默认不设置高度上限 | 正整数，选择高度不超过该值的源视频格式，不进行缩放 |
| `--output-dir` | 可选，默认项目内 `work/downloads` | 输出目录，可使用相对路径或绝对路径；不存在时创建 |
| `-h / --help` | 可选开关 | 显示该子命令的用法，不执行下载 |

时间必须满足 `0 <= start_time < end_time`，单位为秒；1:20:00 写为 `4800`，不写成时间字符串。若指定区间，开始、结束参数必须同时提供。

`section` 必须有时间区间。`full --decode` 省略两个时间参数时解码整个视频，填写两个时间参数时只解码所选区间。`full` 的时间参数必须与 `--decode` 一起使用，不影响完整下载本身。

`--max-height` 依赖网站提供的格式尺寸信息；无匹配格式或尺寸未知时可能失败，不会静默选择更高分辨率。未设置时由 yt-dlp 选择最佳视频格式。

### 4.2 已有本地视频解码：local

无需重新下载，不填写时间参数时解码整个本地文件：

```powershell
.\.venv\Scripts\python.exe main.py local "本地视频完整路径.mp4"
```

指定区间，只读取已有视频的 2–5 秒：

```powershell
.\.venv\Scripts\python.exe main.py local "本地视频完整路径.mp4" --start-time 2 --end-time 5
```

读取已下载的片段，默认解码整个片段，无需手动填写片段时长：

```powershell
.\.venv\Scripts\python.exe main.py local "本地片段完整路径.mp4"
```

| 参数 | 是否必填 / 默认值 | 说明 |
| --- | --- | --- |
| `video_path` | 必填位置参数 | 已有本地视频文件；相对路径以当前终端目录为基准 |
| `--start-time` | 可选，与结束时间成对填写 | 相对于该文件视频流起点的开始秒数，包含起点；两个时间参数均省略时解码全部 |
| `--end-time` | 可选，与开始时间成对填写 | 相对于该文件视频流起点的结束秒数，不包含终点 |
| `-h / --help` | 可选开关 | 显示用法 |

两个时间参数同时省略时解码全部；若指定区间则必须成对填写，支持小数，须满足合法区间且不超过文件时长。在线区间下载后的片段已归零：原视频 4800–4830 秒对应本地约 0–30 秒，不能再次使用 4800–4830 秒解码该文件。

### 4.3 串行批量下载：batch

运行全部成功示例：

```powershell
.\.venv\Scripts\python.exe main.py batch "examples/tasks-all-success.json"
```

运行“前后成功、中间故意 404 失败”示例：

```powershell
.\.venv\Scripts\python.exe main.py batch "examples/tasks.json"
```

使用自己的清单和输出目录：

```powershell
.\.venv\Scripts\python.exe main.py batch "你的任务清单.json" --output-dir "work/custom-batches"
```

| 参数 | 是否必填 / 默认值 | 说明 |
| --- | --- | --- |
| `tasks_file` | 必填位置参数 | UTF-8 JSON 清单路径；格式见第 5.1 节；不是下载后生成的 `results.json` |
| `--output-dir` | 可选，默认项目内 `work/downloads` | 批次输出根目录，每次在其下创建独立 `batch-<标识>` 目录 |
| `-h / --help` | 可选开关 | 显示用法 |

`batch` 只下载文件，不接收 `--decode`、`--mode` 或命令行时间参数。每个任务的下载模式、区间和高度上限填写在 JSON 中。单项运行失败时继续后续任务；结果路径在控制台最后输出。

### 4.4 下载结果统一解码：decode-batch

**完整下载后解码全部内容：**

先使用第 4.3 节的 `batch` 下载，清单中的任务填写 `"mode": "full"`；再运行：

```powershell
.\.venv\Scripts\python.exe main.py decode-batch "results.json完整路径"
```

默认解码每个成功文件的全部内容。既适用于完整视频，也适用于已下载好的区间片段；不会重新下载，清单只包含一个任务时也可使用。

**完整下载后仅解码所需区间，同时处理批次中的区间片段：**

```powershell
.\.venv\Scripts\python.exe main.py decode-batch "results.json完整路径" --full-start-time 2 --full-end-time 5
```

| 参数 | 是否必填 / 默认值 | 说明 |
| --- | --- | --- |
| `results_file` | 必填位置参数 | `batch` 生成的下载汇总文件；不能填写下载前的 `tasks.json` |
| `--full-start-time` | 可选，两个时间参数均省略时解码全部 | 所有成功 full 任务的解码开始秒数，相对于各自文件起点 |
| `--full-end-time` | 可选，与开始时间成对填写 | full 任务的解码结束秒数，不包含终点；每个文件都须覆盖这个区间 |
| `-h / --help` | 可选开关 | 显示用法 |

两个 `--full-*` 参数只影响 full 任务；section 任务始终解码整个已下载片段。填写时必须成对，支持小数，满足合法区间。用户可以选择短区间、长区间或全部内容，程序不设固定时长上限。

下载状态不是 success 的项自动跳过；成功项的文件缺失、损坏或解码越界时记录为本次解码失败，并继续后续项。文件名不必叫 results.json，但内容必须是 batch 生成的汇总结构。

### 4.5 帮助、PyCharm 与调试入口

查看所有主命令的参数：

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py download --help
.\.venv\Scripts\python.exe main.py local --help
.\.venv\Scripts\python.exe main.py batch --help
.\.venv\Scripts\python.exe main.py decode-batch --help
```

PyCharm 运行配置的脚本选择 `main.py`，参数完整填写为：

```text
download "https://www.w3schools.com/html/mov_bbb.mp4" --mode section --start-time 2 --end-time 5 --decode
```

保留 `decoder.py` 用于单独排查本地解码问题。不带时间参数时同样解码整个文件：

```powershell
.\.venv\Scripts\python.exe decoder.py "本地视频完整路径.mp4"
```

指定区间时输出该区间 Tensor：

```powershell
.\.venv\Scripts\python.exe decoder.py "本地视频完整路径.mp4" --start-time 2 --end-time 5
```

需要单独检查第一帧时，Python 代码仍可调用 `read_first_frame(path)`。该调试入口接收必填位置参数 `video_path`、可选成对参数 `--start-time / --end-time`，以及 `-h / --help`。日常使用统一入口 `main.py`。

## 5. 任务清单、输出与 Python 使用

### 5.1 下载任务清单

清单使用 UTF-8 JSON，支持 Windows UTF-8 BOM。以下示例同时下载一个完整视频和一个长视频片段；也可以只保留一个任务：

```json
{
  "tasks": [
    {
      "id": "short-full",
      "url": "https://www.w3schools.com/html/mov_bbb.mp4",
      "mode": "full"
    },
    {
      "id": "long-section",
      "url": "https://www.bilibili.com/video/BV1ahdeBxEC3/",
      "mode": "section",
      "start_time": 4800,
      "end_time": 4830,
      "max_height": 360
    }
  ]
}
```

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `tasks` | 必填 | 非空任务数组；顶层只接受此字段 |
| `id` | 每项必填 | 唯一的非空字符串，用于在汇总中识别任务 |
| `url` | 每项必填 | 单个视频的 HTTP(S) 网页 URL 或媒体直链 |
| `mode` | 每项必填 | 明确填写 `full` 或 `section`，没有隐含默认模式 |
| `start_time` | section 必填 | 原视频开始秒数，JSON 数值，不是字符串 |
| `end_time` | section 必填 | 原视频结束秒数，须大于开始时间 |
| `max_height` | 可选 | 正整数，含义与 download 的同名参数相同 |

full 任务不填写时间字段；其解码区间在 decode-batch 阶段选择。不接受未定义字段。标准 JSON 使用双引号，不支持注释或末尾多余逗号。两份现成示例的详细说明见 [examples/README.md](examples/README.md)。

### 5.2 输出文件与时间轴

单视频下载打印最终文件路径。批量下载的输出结构如下；task 编号对应清单顺序：

```text
work/downloads/batch-<本次运行标识>/
├── task-001/
│   └── video-<唯一标识>.mp4
├── task-002/
│   └── video-<唯一标识>.mp4
└── results.json
```

full 文件的扩展名取决于源格式。results.json 保存任务参数、逐项状态、成功文件路径、字节大小及失败原因；section 成功项还记录实际片段时长、帧数和尺寸。

| 来源 | 解码请求示例 | Tensor 时间戳 |
| --- | --- | --- |
| 完整视频 | 原视频 2–5 秒 | 相对于完整视频流起点，本例 2–4.96 秒 |
| 下载好的 2–5 秒片段 | 本地片段 0–3 秒 | 相对于片段起点，本例 0–2.96 秒 |
| 下载好的 4800–4830 秒片段 | 本地片段约 0–30 秒 | 相对于片段起点；原请求区间单独记录 |

示例视频为 25 FPS：完整 10 秒对应 `(250, 3, 176, 320)`，3 秒对应 `(75, 3, 176, 320)`，dtype 为 `torch.uint8`、device 为 CPU。`[T, C, H, W]` 分别表示帧数、RGB 通道数、高、宽；uint8 像素范围为 0–255。

results.json 中成功文件通常使用绝对路径；移动文件后需修正 path。批量解码也接受相对视频路径，基准为汇总文件所在目录。

### 5.3 已验证的长视频区间示例

只下载原视频 1:20:00–1:20:30，选择最高 360p 格式：

```powershell
.\.venv\Scripts\python.exe main.py download "https://www.bilibili.com/video/BV1ahdeBxEC3/" --mode section --start-time 4800 --end-time 4830 --max-height 360
```

本机已验证得到 H.264 MP4、640×360、30 秒、900 帧。网站返回的格式可能随访问条件改变。该命令只下载并验证首末帧，不将整个片段创建为 Tensor。

### 5.4 Python 中取得 Tensor

本地文件默认整段解码：

```python
from pathlib import Path
from decoder import read_video_clip

clip = read_video_clip(Path("本地视频完整路径.mp4"))
video_tensor = clip.data
```

需要区间时，传入 `start_time=2, end_time=5`；两个时间参数同时省略则处理整个文件。

单视频下载并解码所选区间：

```python
from pathlib import Path
from pipeline import video_url_to_tensor

result = video_url_to_tensor(
    url="https://www.w3schools.com/html/mov_bbb.mp4",
    start_time=2,
    end_time=5,
    output_dir=Path("work/downloads"),
    mode="section",
)
video_tensor = result.clip.data
timestamps = result.clip.pts_seconds
local_video = result.download.path
```

该 API 的 `mode` 默认是 `section`；明确传入 `mode="full"` 则先完整下载，再解码传入区间，时间戳相对于完整视频流起点。完整下载后整段解码时，将两个时间参数同时传为 `None`：

```python
result = video_url_to_tensor(
    url="https://www.w3schools.com/html/mov_bbb.mp4",
    start_time=None,
    end_time=None,
    output_dir=Path("work/downloads"),
    mode="full",
)
video_tensor = result.clip.data
```

读取下载汇总，逐条取得 Tensor，默认解码全部内容：

```python
from pathlib import Path
from batch_decoder import iter_decoded_results

for item in iter_decoded_results(
    Path("替换为下载结果的results.json完整路径"),
):
    if item.clip is not None:
        video_tensor = item.clip.data
        print(item.entry.id, video_tensor.shape)
        # 在这里使用当前 Tensor。
        del video_tensor
    elif item.error:
        print(item.entry.id, item.error)
    del item
```

如果只需 full 视频的部分内容，向 `iter_decoded_results` 选填 `full_start_time=2, full_end_time=5`。生成器通过 yield 一次交出一项，调用方继续迭代时才处理下一项，类似 Java Iterator；避免将所有返回项放进列表而累积多个视频的 Tensor。

## 6. 当前行为、边界与验证

### 在线区间下载的支持范围

- yt-dlp 解析网页中的媒体地址，FFmpeg 打开远程输入时设置开始秒数和读取时长，不先下载完整文件再切片。
- 当前仅接受解析后的 HTTP(S) 媒体直链。下载前要求服务器对单字节 Range 探测返回 HTTP 206 和匹配的 Content-Range。B 站样例解析后得到可范围访问的 HTTP 媒体文件；m3u8/mpd 清单式来源尚未验证，会明确拒绝。
- 不支持 Range、下载失败或输出验证失败时直接报错，不自动改为完整下载。Range 支持也不保证传输字节数恰好等于片段大小，还可能读取文件头、索引、前置关键帧或缓冲数据。
- 区间输出重新编码为 H.264 / yuv420p / MP4，CRF 18、fast preset，不保留音频。转码消耗 CPU 并改变像素值。
- 片段时间轴归零。重编码后每帧的本地时间戳不等于对原始帧时间的绝对精确还原。
- TorchCodec 验证首末帧可解码；实际时长与请求的差异不得超过 0.1 秒或一个边界帧时长（取较大值）。这是异常检查，不承诺任意小数秒边界绝对准确。
- 已知源时长越界时提前拒绝；源时长未知或不准确时依靠输出检查拒绝明显异常。空片段、解码失败或时长异常不保留为成功文件。
- 只处理已有单视频，不支持直播、尚未开始的视频或纯播放列表。具体网站可能受访问条件影响，需要用目标 URL 验证。

### 本地解码与内存

按真实帧起始时间戳选择 `[start_time, end_time)` 内的帧，支持可变帧率，不用平均 FPS 推算帧号或复制帧。区间内没有帧起点时会报错，越界不自动截短；帧不可分割，最后一帧的持续时间可能跨过终点。

单个目标视频或片段整体解码到内存。批量 CLI 处理完一项后释放引用，再处理下一项；长视频的内存需求随帧数和分辨率增加，是否能全部解码取决于可用内存。原分辨率、帧数和 uint8 像素保持不变，不自动采样、缩放或归一化。

Tensor 在内存中生成，命令行打印摘要，Python 调用可直接使用实际 Tensor；当前不自动保存 `.pt` 文件，程序退出后内存中的 Tensor 不再保留。

### 批次、文件与错误处理

- 先检查整份下载清单的结构、URL 格式、时间参数和重复 id；配置错误时整批不开始下载。网站访问及实际时长问题在各项执行时判断。
- 视频任务串行处理。单项下载或解码失败记录错误后继续；下载状态非 success 的项在批量解码时跳过。
- 下载汇总在开始时及每项执行前后写入 results.json，状态 completed 表示本批次已处理完，不表示每项都成功。批量解码不修改该文件，控制台单独汇总解码结果。
- 成功下载文件保留，每次使用独立文件或批次目录，不覆盖旧结果。Tensor 解码失败时已下载视频保留，错误中提供路径。
- 普通失败或正常 Ctrl+C 取消时清理当前下载临时文件；强制结束进程或断电可能留下临时目录，汇总可能停在 running。
- Ctrl+C 停止后续任务。下载阶段保留已成功文件及汇总；解码阶段输出已完成项的计数。
- 网络读取等待超时设置为 20 秒，这不是整个任务的总时间上限。
- 不提供自动断点续跑、缓存复用、并发下载、手动分片或视频拼接。需重试时可将失败任务参数复制到新清单。

| 退出码 | 含义 |
| --- | --- |
| 0 | 当前操作成功；批量解码允许跳过原下载失败项，全部跳过时也会输出计数 |
| 1 | 处理失败、清单内容或 JSON 无效，或本次批量处理含失败项 |
| 2 | 命令行参数缺失或组合错误 |
| 130 | 用户手动取消 |

### 自动测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

现有 29 个测试方法覆盖本地解码、真实 FFmpeg HTTP Range 区间下载、片段时间轴、对应帧转码误差、范围请求与传输量、异常片段拒绝和清理；还覆盖批量预检查、失败继续、取消、重复运行，以及完整视频默认整段 / 可选区间、损坏或缺失文件、相对路径、UTF-8 BOM、Tensor 引用释放和 CLI 集成。

自动测试在 work 内临时生成视频，结束后清理，无需访问外部网站；FFmpeg 须在 PATH 中。公网示例依赖网站及网络可用性。

## 版本记录

### MVP v0.21

- 修复 `main.py local` 强制要求起止时间的问题：省略两个时间参数时，默认解码整个本地视频。
- 修复 `decoder.py` 不填时间时只读取第一帧的行为：现在默认解码整个文件。
- `read_video_clip(path)` 支持省略时间参数，直接返回整段视频 Tensor；指定成对时间时仍按区间解码。
- 明确默认规则：完整下载、下载后整段解码和批量整段解码均无需填写时间；主动选择 section 区间下载时仍须填写起止时间。只填写一个时间参数会报错。
- 补充默认整段解码的像素与命令行回归测试，29 项测试通过；同步更新使用示例和参数说明。

### MVP v0.2

- 新增在线指定时间区间下载，验证远程范围访问、实际片段时长与可解码性。
- 新增 JSON 清单驱动的串行批量下载，支持混合 full / section、失败继续及结果汇总。
- 新增读取 results.json 的批量解码，默认解码完整文件，可为 full 项统一指定区间。
- 单视频完整下载后支持默认整段解码，或按用户指定区间解码。
- 统一 download / local / batch / decode-batch 命令入口，保留独立解码调试入口。
- 完善参数检查、错误处理、临时文件清理及测试；补全环境、快速开始、命令参数和输出说明。

### MVP v0.1

- 跑通在线视频 URL → yt-dlp 完整下载 → TorchCodec 解码的最小链路。
- 支持本地指定时间区间提取 PyTorch Tensor，输出 shape、dtype 和基本视频信息。
- 提供 Windows 环境安装说明及基础验证。

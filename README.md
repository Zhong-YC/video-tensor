# video-tensor

将在线视频转换为可供后续 PyTorch 处理的视频 Tensor。

处理流程：**输入 URL → yt-dlp 完整下载视频 → TorchCodec 提取指定时间区间 → 返回 PyTorch Tensor**。

## Quick Start

前提：已安装 Python 3.12 x64、FFmpeg 8.1.2 shared，并将 FFmpeg 的 `bin` 加入 PATH（详见第 2 节）。在项目根目录打开 PowerShell，依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py "https://www.w3schools.com/html/mov_bbb.mp4" --start-time 2 --end-time 5
```

预期显示“时间区间解码成功”、`shape=(75, 3, 176, 320)` 和 `torch.uint8`。已有配置好的 `.venv` 时直接执行最后一条即可。

## 项目目录

```text
video-tensor/
├── main.py              # 统一命令行入口
├── pipeline.py          # 串联下载与解码
├── downloader.py        # 完整视频下载
├── decoder.py           # 视频帧与时间区间解码
├── tests/               # 自动测试源码
├── requirements.txt     # Python 依赖版本
├── README.md            # 安装与运行说明
└── .gitignore           # 本地文件排除规则
```

`.venv/` 和 `work/` 是本地运行时目录，不属于交付源码。

## 1. Python 版本

本项目已在 **Windows 11 x64 + Python 3.12.2** 上验证，建议使用 Python 3.12 的 64 位版本，并为项目创建独立虚拟环境。

已验证的主要依赖：

| 组件 | 版本 |
| --- | --- |
| PyTorch | 2.10.0+cpu |
| TorchCodec | 0.10.0 |
| yt-dlp | 2026.8.19 |
| NumPy | 2.5.3 |
| FFmpeg | 8.1.2 shared |

当前使用 CPU 解码，不需要安装 CUDA。Python 依赖版本已写入 `requirements.txt`；其他操作系统尚未验证。

## 2. FFmpeg 安装要求

安装 **FFmpeg 8.1.2 的 Windows x64 shared 构建**。可以从 [FFmpeg 下载页](https://ffmpeg.org/download.html) 进入 Windows 构建下载站，选择包含 `shared` 的对应版本，例如 `ffmpeg-8.1.2-full_build-shared.7z`。

解压后，将其 `bin` 目录加入 Windows 用户环境变量 `Path`。该目录应包含：

- `ffmpeg.exe` 和 `ffprobe.exe`。
- `avcodec-62.dll`、`avformat-62.dll`、`avutil-60.dll` 等动态库；应保留整套 shared 文件。

例如，若文件位于 `D:\Tools\ffmpeg-8.1.2-full_build-shared\bin\ffmpeg.exe`，则加入 Path 的是 `D:\Tools\ffmpeg-8.1.2-full_build-shared\bin`。

保存后重新打开终端；使用 PyCharm 内置终端时，可能需要重启 PyCharm。执行：

```powershell
ffmpeg -version
ffprobe -version
```

两条命令都应输出版本信息。TorchCodec 需要 FFmpeg 动态库，只有命令行程序的构建不够；`pip install ffmpeg` 不能替代这一步。请使用上述已验证版本组合。

## 3. pip 安装命令

在项目根目录打开 PowerShell，确认 `python --version` 为 Python 3.12，然后依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

已有可用 `.venv` 时，无需重新创建。依赖文件已配置 PyTorch CPU 安装源。

验证依赖：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; from torchcodec.decoders import VideoDecoder; print('PyTorch:', torch.__version__); print('TorchCodec import OK')"
```

预期显示 `PyTorch: 2.10.0+cpu` 和 `TorchCodec import OK`。这一步验证依赖加载；完整下载与解码请执行第 5 节示例。

### 使用 PyCharm 时的可选配置

完成上述安装后，如果通过 PyCharm 的运行或调试按钮执行程序，请将项目解释器指向已有的虚拟环境：

1. 按 `Ctrl + Alt + S` 打开设置，搜索“Python 解释器”或 `Python Interpreter`。
2. 打开解释器页面，检查当前路径是否为项目下的 `.venv\Scripts\python.exe`；若已正确，无需更改。
3. 如需修改，选择“添加解释器 → 添加本地解释器”，选择已有环境（不同版本的菜单名称可能略有区别）。
4. 浏览并选择 `<项目目录>\.venv\Scripts\python.exe`，点击“应用 / 确定”。不要重复创建虚拟环境。

例如，项目位于 `D:\New-CodeDesktop\video-tensor` 时，应选择 `D:\New-CodeDesktop\video-tensor\.venv\Scripts\python.exe`；其他机器请使用自己的项目路径。

这一步只配置 PyCharm 使用哪个 Python，不会重新安装依赖。设置项目解释器也不保证已有终端自动切换环境；终端中仍可用 `python -c "import sys; print(sys.executable)"` 检查实际路径。

如果始终按本文命令直接调用 `.\.venv\Scripts\python.exe`，则无需配置 PyCharm 解释器，也无需先激活虚拟环境。

## 4. 怎么运行

统一入口为 `main.py`，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe main.py "URL" --start-time 2 --end-time 5
```

将 `URL` 替换为实际视频地址，并保留引号。

| 参数 | 说明 |
| --- | --- |
| `URL` | 单个在线视频网页地址或视频直链 |
| `--start-time` | 开始时间，单位为秒，包含该时刻 |
| `--end-time` | 结束时间，单位为秒，不包含该时刻 |
| `--output-dir` | 可选，本地视频保存目录，默认项目内的 `work/downloads` |

起止时间必须同时提供，满足 `0 <= start_time < end_time <= 视频流时长`，支持小数。

如果已激活项目虚拟环境，或者 PyCharm 终端中的 `python` 已指向项目的 `.venv`，可使用简写：

```powershell
python main.py "URL" --start-time 2 --end-time 5
```

可用 `python -c "import sys; print(sys.executable)"` 确认解释器路径。在 PyCharm 中运行时，项目解释器应选择项目下的 `.venv\Scripts\python.exe`，脚本参数填写 `"URL" --start-time 2 --end-time 5`。

## 5. 示例

使用公开视频提取 2–5 秒：

```powershell
.\.venv\Scripts\python.exe main.py "https://www.w3schools.com/html/mov_bbb.mp4" --start-time 2 --end-time 5
```

该示例已完成实际下载、解码和重复运行验证。关键输出如下，下载文件名每次不同：

```text
下载成功
标题：mov_bbb
大小：788,493 字节
时间区间解码成功
请求区间（秒）：[2.0, 5.0)
视频流时长（秒）：10.0
编码：h264；平均帧率：25.0
Tensor shape [T, C, H, W]：(75, 3, 176, 320)
Tensor dtype：torch.uint8；device：cpu
首/末帧相对时间戳（秒）：2.000000 / 4.960000
像素 Tensor 内存：12.08 MiB
```

其中 `T` 是帧数，`C` 是 RGB 三个颜色通道，`H/W` 是图像高宽。示例返回 75 帧，每帧大小为 176×320，像素类型为 `uint8`，取值为 0–255。

如需在 Python 代码中直接使用结果：

```python
from pathlib import Path
from pipeline import video_url_to_tensor

result = video_url_to_tensor(
    url="https://www.w3schools.com/html/mov_bbb.mp4",
    start_time=2,
    end_time=5,
    output_dir=Path("work/downloads"),
)
video_tensor = result.clip.data  # 实际的 torch.Tensor，形状为 [T, C, H, W]
timestamps = result.clip.pts_seconds
local_video = result.download.path
```

## 6. 当前 MVP 的行为

### 下载与输出

- 先完整下载视频，再按时间区间解码，只返回选中片段对应的像素 Tensor。
- 优先下载独立视频流，否则使用含音频的视频；不强制 MP4、不转码、不合并音轨，下载文件可能没有声音。
- 每次运行创建新的本地视频文件，不覆盖旧文件、不复用缓存。成功下载的完整视频保留在输出目录中。
- 下载临时文件在正常结束、一般异常或正常处理 Ctrl+C 时自动清理；强制结束进程或断电可能留下 `download-*` 目录。
- 命令行输出 Tensor 摘要；实际 Tensor 在进程内生成，不自动保存 `.pt`，也不生成独立的裁剪视频文件。Python 调用者通过 `result.clip.data` 获取 Tensor。

### 时间规则与错误处理

- 时间以视频流首帧为零点，只返回帧起始时间戳位于 `[start_time, end_time)` 的帧；返回的时间戳也相对于视频流起点，而非片段起点。
- 按真实时间戳选择帧，支持可变帧率，不用平均 FPS 推算帧号。若起点落在两帧之间，选取后面开始的帧。
- 帧持续时间保留原值，最后一帧的显示区间可能跨过结束时间。区间内没有任何帧起始时间戳时会报错。
- 对无效 URL、非法时间、下载失败、文件无效或解码失败给出错误提示。超出时长不会自动截短；下载成功后解码失败，错误中会提供保留的本地文件路径。
- 正常退出码为 0，处理失败为 1，命令行参数格式错误或缺失为 2，手动取消为 130。

### 范围与验证

当前处理单个已有视频，不支持纯播放列表或直播。已验证上述公开视频直链；具体网站可能受登录、地区限制、网站变化或额外运行时要求影响，需使用目标 URL 验证。

Tensor 保留原始分辨率和帧数，不自动采样、缩放、归一化或训练模型。当前一次返回整个片段，建议先用短视频、短区间验证内存开销。示例只有 10 秒，测试 80–90 秒需换用更长的视频。

自动测试覆盖恒定/可变帧率、非零起始时间、时间边界及非法参数，并与按帧号读取的像素结果比对。测试在本地临时生成小视频，无需联网，需 FFmpeg 在当前终端 PATH 中：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 保留的调试入口

日常使用 `main.py` 完成全部流程；单独功能保留用于定位问题和复用已下载的视频，无需每次重新下载。

```powershell
# 仅下载：省略两个时间参数
.\.venv\Scripts\python.exe main.py "URL"

# 本地视频读取第一帧
.\.venv\Scripts\python.exe decoder.py "本地视频路径"

# 本地视频提取指定片段
.\.venv\Scripts\python.exe decoder.py "本地视频路径" --start-time 2 --end-time 5
```

代码按职责分开，统一入口负责调用：

| 文件 | 职责 |
| --- | --- |
| `main.py` | 解析命令行参数，输出结果与错误 |
| `pipeline.py` | 串联下载和片段解码，返回结果 |
| `downloader.py` | 使用 yt-dlp 下载完整视频 |
| `decoder.py` | 使用 TorchCodec 读取帧和时间区间 |
| `tests/` | 自动验证时间选择和参数检查 |

当前状态：MVP v0.1 / 已验证 Windows 11。

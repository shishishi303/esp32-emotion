"""把 .tflite 转成 C 头文件（xxd -i 风格）。

    uv run python scripts/export_c_header.py

生成的 firmware/pupil.h 可以直接覆盖固件里的
components/emotion_recognition/pupil.h，数组名保持一致，C++ 侧无需改动。
"""

import argparse
import sys
from pathlib import Path

# 允许从项目根目录以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion.config import HEADER_ARRAY_NAME, HEADER_PATH, TFLITE_PATH  # noqa: E402

HEADER_TEMPLATE = """\
// 由 scripts/export_c_header.py 自动生成，请勿手工编辑。
// 源文件: {source}
// 大小:   {size} 字节
// SHA256: {sha256}

#pragma once

const unsigned int {array}_len = {size};
const unsigned char {array}[] = {{
{body}
}};
"""


def format_bytes(data: bytes, per_line: int = 12) -> str:
    lines = []
    for offset in range(0, len(data), per_line):
        chunk = data[offset : offset + per_line]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    return "\n".join(lines)


def build_header(data: bytes, source_name: str, array_name: str) -> str:
    import hashlib

    return HEADER_TEMPLATE.format(
        source=source_name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest().upper(),
        array=array_name,
        body=format_bytes(data),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="导出 C 头文件")
    parser.add_argument("--tflite", default=str(TFLITE_PATH))
    parser.add_argument("--output", default=str(HEADER_PATH))
    parser.add_argument("--array-name", default=HEADER_ARRAY_NAME)
    args = parser.parse_args(argv)

    source = Path(args.tflite)
    if not source.is_file():
        raise SystemExit(f"找不到 tflite 文件: {source}（先跑 emotion.quantize）")

    data = source.read_bytes()
    header = build_header(data, source.name, args.array_name)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header, encoding="utf-8", newline="\n")

    print(f"已生成 {output}")
    print(f"  数组名: {args.array_name}[]")
    print(f"  大小:   {len(data)} 字节 ({len(data) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

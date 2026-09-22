#!/usr/bin/env python3
"""运行全部教程脚本，并把脚本输出的 Markdown 正文写成每课教材。"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def render(run_script: Path) -> None:
    lesson_dir = run_script.parent
    result = subprocess.run(
        [str(run_script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    relative_script = run_script.relative_to(ROOT)
    readme_first_line = (lesson_dir / "README.md").read_text(encoding="utf-8").splitlines()[0]
    title = readme_first_line.removeprefix("# ")
    markdown = (
        f"# {title}\n\n"
        "> 本文件由教程脚本自动生成。请修改源脚本后重新渲染，不要手工编辑。\n\n"
        f"生成来源：`{relative_script}`\n\n"
        f"{result.stdout.strip()}\n"
    )
    output_path = lesson_dir / "OUTPUT.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"已生成：{output_path.relative_to(ROOT)}")


def main() -> None:
    run_scripts = sorted((ROOT / "lessons").glob("**/run.sh"))
    if not run_scripts:
        raise SystemExit("没有找到 lessons/**/run.sh")
    for run_script in run_scripts:
        render(run_script)


if __name__ == "__main__":
    main()

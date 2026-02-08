"""
Video Format Conversion

Converts uploaded videos from any format to mp4 using FFmpeg
bundled via imageio-ffmpeg. No system FFmpeg install required.

Supported input formats: mov, avi, webm, mkv, wmv, flv, m4v, ts, and more.
Output: H.264 mp4 compatible with OpenCV and browser playback.
"""

import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".webm", ".mkv", ".wmv",
    ".flv", ".m4v", ".ts", ".mts", ".3gp", ".ogv",
}


def needs_conversion(file_path: str) -> bool:
    """Check if a video file needs conversion to mp4.

    Args:
        file_path: Path to the video file.

    Returns:
        True if the file is not already mp4.
    """
    return Path(file_path).suffix.lower() != ".mp4"


def convert_to_mp4(input_path: str, output_path: str | None = None) -> str:
    """Convert a video file to mp4 (H.264 + AAC).

    Args:
        input_path: Path to the source video file.
        output_path: Path for the converted file. If None,
            replaces the extension with .mp4 in the same directory.

    Returns:
        Path to the converted mp4 file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If FFmpeg conversion fails.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".mp4")
    else:
        output_path = Path(output_path)

    if input_path.suffix.lower() == ".mp4" and input_path == output_path:
        return str(input_path)

    ffmpeg = get_ffmpeg_exe()

    cmd = [
        ffmpeg,
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(
            f"FFmpeg conversion failed (code {result.returncode}): {error_msg}"
        )

    if not output_path.exists():
        raise RuntimeError(f"Conversion produced no output: {output_path}")

    return str(output_path)


def ensure_mp4(file_path: str) -> str:
    """Ensure a video file is in mp4 format, converting if necessary.

    If already mp4, returns the original path unchanged.
    If not, converts to mp4 alongside the original and returns the new path.

    Args:
        file_path: Path to the video file.

    Returns:
        Path to an mp4 version of the video.
    """
    if not needs_conversion(file_path):
        return file_path

    mp4_path = str(Path(file_path).with_suffix(".mp4"))
    return convert_to_mp4(file_path, mp4_path)

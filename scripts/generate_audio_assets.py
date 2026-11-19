"""
DriveGuard — Offline Audio Asset Generator (Phase 8).

Generates pre-rendered offline WAV audio assets for:
1. Voice announcements (ATTENTION, WARNING, CRITICAL, RECOVERY) using offline speech synthesis.
2. Warning tones (Attention Chime, Warning Alert, Critical Danger Pulse) using numpy sinusoidal synthesis.

Zero cloud or internet dependencies. Pre-rendered assets are stored in:
`backend/app/alerts/assets/`
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import struct
import subprocess
import wave

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("driveguard.audio_gen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "backend" / "app" / "alerts" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Tone Synthesis (Pure NumPy + Wave)
# -----------------------------------------------------------------------------

def generate_tone_wav(
    filepath: Path,
    frequencies: list[float],
    durations_ms: list[float],
    silence_ms: list[float],
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> None:
    """
    Synthesize harmonic audio tone pulses with smooth attack and decay envelopes.
    """
    samples: list[float] = []

    for freq, dur, sil in zip(frequencies, durations_ms, silence_ms):
        n_samples = int((dur / 1000.0) * sample_rate)
        t = np.linspace(0, dur / 1000.0, n_samples, endpoint=False)
        
        # Fundamental sine wave + subtle 2nd harmonic
        wave_data = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
        
        # Smooth attack (10ms) and exponential decay envelope
        attack_len = min(int(0.010 * sample_rate), n_samples // 4)
        if attack_len > 0:
            attack_env = np.linspace(0.0, 1.0, attack_len)
            wave_data[:attack_len] *= attack_env
            
        decay_env = np.exp(-3.0 * t / (dur / 1000.0))
        wave_data *= decay_env
        
        # Normalize and scale
        max_val = np.max(np.abs(wave_data)) or 1.0
        wave_data = (wave_data / max_val) * amplitude
        samples.extend(wave_data)

        # Silence gap
        if sil > 0:
            n_silence = int((sil / 1000.0) * sample_rate)
            samples.extend(np.zeros(n_silence, dtype=np.float32))

    audio_array = np.array(samples, dtype=np.float32)
    int16_data = np.int16(audio_array * 32767.0)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    logger.info("Generated tone: %s (%.2fs)", filepath.name, len(audio_array) / sample_rate)


def generate_tones() -> None:
    """Generate all warning tones."""
    # 1. Attention Tone: Gentle rising dual chime (587 Hz D5 -> 880 Hz A5)
    generate_tone_wav(
        filepath=ASSETS_DIR / "attention_tone.wav",
        frequencies=[587.33, 880.0],
        durations_ms=[150.0, 280.0],
        silence_ms=[30.0, 0.0],
        amplitude=0.45,
    )

    # 2. Warning Tone: Double caution chime (750 Hz pair)
    generate_tone_wav(
        filepath=ASSETS_DIR / "warning_tone.wav",
        frequencies=[750.0, 750.0],
        durations_ms=[180.0, 220.0],
        silence_ms=[60.0, 0.0],
        amplitude=0.55,
    )

    # 3. Critical Tone: Urgent triple pulse (1000 Hz sharp)
    generate_tone_wav(
        filepath=ASSETS_DIR / "critical_tone.wav",
        frequencies=[1000.0, 1000.0, 1000.0],
        durations_ms=[90.0, 90.0, 140.0],
        silence_ms=[40.0, 40.0, 0.0],
        amplitude=0.70,
    )


# -----------------------------------------------------------------------------
# Voice Synthesis (Offline SAPI on Windows)
# -----------------------------------------------------------------------------

VOICE_ANNOUNCEMENTS = {
    "attention_voice.wav": "Please stay alert. You may be showing early signs of fatigue.",
    "warning_voice.wav": "You are showing signs of drowsiness. Please take a break.",
    "critical_voice.wav": "Danger! You may be falling asleep. Pull over safely as soon as possible.",
    "recovery_voice.wav": "Your alert status has returned to normal.",
}


def generate_voices() -> None:
    """Generate voice announcements via Windows System.Speech."""
    for filename, text in VOICE_ANNOUNCEMENTS.items():
        out_path = ASSETS_DIR / filename
        ps_cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"try {{ $s.SelectVoice('Microsoft Zira Desktop') }} catch {{ }}; "
            f"$s.SetOutputToWaveFile('{out_path.as_posix()}'); "
            f"$s.Speak('{text}'); "
            f"$s.Dispose()"
        )
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                check=True,
            )
            if out_path.is_file() and out_path.stat().st_size > 100:
                logger.info("Generated voice: %s (%d bytes)", filename, out_path.stat().st_size)
            else:
                logger.error("Failed to generate voice %s: empty file", filename)
        except Exception as e:
            logger.error("Error generating voice %s via SAPI: %s", filename, e)


def main() -> None:
    logger.info("Generating Phase 8 offline audio assets in: %s", ASSETS_DIR)
    generate_tones()
    generate_voices()
    logger.info("Audio asset generation complete.")


if __name__ == "__main__":
    main()

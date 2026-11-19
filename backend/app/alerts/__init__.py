"""
DriveGuard — Alert State Machine, Warning Dispatcher, and Audio System (Phase 8).

Exports:
- State & Transition models: `AlertState`, `AlertTransition`, `map_drowsiness_state_to_alert_state`
- Alert State Manager: `AlertStateManager`
- Dispatcher: `AlertDispatcher`
- Audio System: `AudioWarningEngine`, `AudioBackend`, `SoundDeviceAudioBackend`, `MockAudioBackend`, `AudioQueueItem`
- Audio Settings: `AudioSettingsStore`, `audio_settings_store`
- Structured Logging: `AlertEventLogger`, `AlertLogEntry`
"""

from backend.app.alerts.alert_dispatcher import AlertDispatcher
from backend.app.alerts.alert_state import (
    AlertState,
    AlertTransition,
    map_drowsiness_state_to_alert_state,
)
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.alerts.audio_settings import AudioSettingsStore, audio_settings_store
from backend.app.alerts.audio_warning_engine import (
    AudioBackend,
    AudioQueueItem,
    AudioWarningEngine,
    MockAudioBackend,
    SoundDeviceAudioBackend,
)
from backend.app.alerts.event_logger import AlertEventLogger, AlertLogEntry

__all__ = [
    "AlertDispatcher",
    "AlertEventLogger",
    "AlertLogEntry",
    "AlertState",
    "AlertStateManager",
    "AlertTransition",
    "AudioBackend",
    "AudioQueueItem",
    "AudioSettingsStore",
    "AudioWarningEngine",
    "MockAudioBackend",
    "SoundDeviceAudioBackend",
    "audio_settings_store",
    "map_drowsiness_state_to_alert_state",
]

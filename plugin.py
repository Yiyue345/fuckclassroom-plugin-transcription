from __future__ import annotations

from pathlib import Path

from fuckclassroom.core.plugins import PluginContext, PluginSpec, SettingsPanel


PLUGIN_DIR = Path(__file__).resolve().parent

def setup_services(context: PluginContext):
    from .services import setup_services as setup
    return setup(context)


async def startup(context: PluginContext):
    from .services import startup as hook
    return await hook(context)


async def shutdown(context: PluginContext):
    from .services import shutdown as hook
    return await hook(context)


def build_settings_context(services, request):
    from .environment import build_settings_context as build
    return build(services, request)


def build_routes(context: PluginContext):
    from .routes import build_router
    return build_router(context)


def build_plugin() -> PluginSpec:
    return PluginSpec(
        id="transcription",
        name="本地语音转写",
        order=30,
        requires=("processing",),
        service_factory=setup_services,
        route_factory=build_routes,
        startup=startup,
        shutdown=shutdown,
        template_dir=PLUGIN_DIR / "templates",
        settings_panels=(
            SettingsPanel(
                key="transcription",
                label="语音转写",
                template="transcription_settings.html",
                order=50,
                checkbox_fields=("asr_use_batched", "asr_simplified_chinese"),
                context_factory=build_settings_context,
            ),
        ),
    )


__all__ = ["build_plugin"]

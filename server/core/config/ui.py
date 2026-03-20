from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import clean_text, decode_env_text, env_first_non_empty, normalize_banner_type, parse_json, sanitize_id, to_bool_value


DEFAULT_SOC_PROMPT_TEMPLATE = (
    "SoC, ASIC, chiplet, CPU, GPU, NPU, DSP, ISP, VPU, DPU, MCU, PMU, NoC, interconnect, "
    "AXI, AXI4, AXI-Lite, AHB, APB, ACE, CHI, UCIe, PCIe, CXL, DDR, DDR4, DDR5, LPDDR4, "
    "LPDDR5, HBM, SRAM, ROM, eMMC, UFS, PHY, SerDes, PLL, DLL, RC oscillator, clock, "
    "clock tree, clock gating, reset, async reset, sync reset, power domain, voltage island, "
    "retention, isolation, level shifter, DVFS, AVS, UPF, CPF, RTL, SystemVerilog, Verilog, "
    "VHDL, UVM, testbench, assertion, SVA, lint, SpyGlass, CDC, RDC, STA, MCMM, OCV, AOCV, "
    "POCV, derate, setup, hold, recovery, removal, skew, jitter, uncertainty, timing closure."
)


@dataclass(frozen=True)
class UiConfig:
    app_brand_title: str
    app_brand_tagline: str
    ui_prompt_templates: tuple[dict[str, str], ...]
    ui_banners: tuple[dict[str, Any], ...]


def _parse_simple_ui_banners(raw: str) -> tuple[dict[str, Any], ...]:
    decoded = decode_env_text(raw).strip()
    if not decoded:
        return ()
    banners: list[dict[str, Any]] = []
    entries = [part.strip() for part in decoded.split(";;") if part.strip()]
    for index, entry in enumerate(entries, start=1):
        segments = [segment.strip() for segment in entry.split("|")]
        if not segments:
            continue
        banner_type = "info"
        title = ""
        message = ""
        dismissible = True
        if len(segments) == 1:
            message = segments[0]
        elif len(segments) == 2:
            banner_type, message = segments
        elif len(segments) == 3:
            banner_type, title, message = segments
        else:
            banner_type = segments[0]
            title = segments[1]
            dismissible = to_bool_value(segments[-1], True)
            message = "|".join(segments[2:-1]).strip()
        clean_message = clean_text(message, 2000)
        if not clean_message:
            continue
        banners.append({
            "id": f"banner-{index}",
            "type": normalize_banner_type(banner_type),
            "title": clean_text(title, 200),
            "message": clean_message,
            "dismissible": dismissible,
        })
    return tuple(banners)


def parse_ui_banners() -> tuple[dict[str, Any], ...]:
    raw = env_first_non_empty("APP_UI_BANNERS_TEXT", "APP_UI_BANNERS", "UI_BANNERS", "WEBUI_BANNERS") or ""
    if not raw:
        return ()
    parsed = parse_json(raw)
    if parsed is None:
        parsed_simple = _parse_simple_ui_banners(raw)
        if parsed_simple:
            return parsed_simple
        message = clean_text(decode_env_text(raw), 2000)
        if not message:
            return ()
        return ({"id": "banner-1", "type": "info", "title": "", "message": message, "dismissible": True},)
    items = parsed if isinstance(parsed, list) else [parsed]
    banners: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        fallback_id = f"banner-{index}"
        if isinstance(item, str):
            message = clean_text(item, 2000)
            if message:
                banners.append({"id": fallback_id, "type": "info", "title": "", "message": message, "dismissible": True})
            continue
        if not isinstance(item, dict) or not to_bool_value(item.get("enabled"), True):
            continue
        message = clean_text(item.get("message") or item.get("content") or item.get("text") or item.get("body"), 2000)
        if not message:
            continue
        banners.append({
            "id": sanitize_id(str(item.get("id") or ""), fallback_id),
            "type": normalize_banner_type(item.get("type") or item.get("level")),
            "title": clean_text(item.get("title"), 200),
            "message": message,
            "dismissible": to_bool_value(item.get("dismissible"), True),
        })
    return tuple(banners)


def parse_prompt_templates() -> tuple[dict[str, str], ...]:
    raw = env_first_non_empty("APP_PROMPT_TEMPLATES", "APP_SOC_PROMPT_TEMPLATE") or ""
    if not raw:
        return ({"id": "soc-design", "label": "SoC設計テンプレート", "content": DEFAULT_SOC_PROMPT_TEMPLATE},)
    parsed = parse_json(raw)
    if parsed is None:
        content = decode_env_text(raw).strip()
        return ({"id": "template-1", "label": "Template 1", "content": content},) if content else ()
    items = parsed if isinstance(parsed, list) else [parsed]
    templates: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            content = decode_env_text(item).strip()
            if content:
                templates.append({"id": f"template-{index}", "label": f"Template {index}", "content": content})
            continue
        if not isinstance(item, dict):
            continue
        content = decode_env_text(str(item.get("content") or item.get("prompt") or "")).strip()
        if not content:
            continue
        label = str(item.get("label") or item.get("name") or f"Template {index}").strip() or f"Template {index}"
        templates.append({"id": sanitize_id(str(item.get("id") or ""), f"template-{index}"), "label": label[:40], "content": content})
    return tuple(templates)


def load_ui_config() -> UiConfig:
    return UiConfig(
        app_brand_title=decode_env_text(env_first_non_empty("APP_BRAND_TITLE") or "whistx"),
        app_brand_tagline=decode_env_text(env_first_non_empty("APP_BRAND_TAGLINE") or "高精度リアルタイム文字起こし"),
        ui_prompt_templates=parse_prompt_templates(),
        ui_banners=parse_ui_banners(),
    )

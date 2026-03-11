from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    ws_path: str
    transcripts_dir: Path
    openai_api_key: str
    openai_base_url: str | None
    asr_model: str
    summary_api_key: str
    summary_base_url: str | None
    summary_model: str
    summary_temperature: float
    summary_input_max_chars: int
    summary_system_prompt: str
    summary_prompt_template: str
    proofread_api_key: str
    proofread_base_url: str | None
    proofread_model: str
    proofread_temperature: float
    proofread_input_max_chars: int
    proofread_system_prompt: str
    proofread_prompt_template: str
    diarization_enabled: bool
    diarization_hf_token: str
    diarization_model: str
    diarization_device: str
    diarization_sample_rate: int
    diarization_num_speakers: int
    diarization_min_speakers: int
    diarization_max_speakers: int
    diarization_work_dir: Path
    diarization_keep_chunks: bool
    ffmpeg_bin: str
    asr_preprocess_enabled: bool
    asr_preprocess_sample_rate: int
    asr_overlap_ms: int
    default_language: str
    default_prompt: str
    default_temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    context_recent_lines: int
    context_term_limit: int
    max_queue_size: int
    max_chunk_bytes: int
    app_brand_title: str
    app_brand_tagline: str
    ui_prompt_templates: tuple[dict[str, str], ...]
    ui_banners: tuple[dict[str, Any], ...]


def _env_first_non_empty(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def _to_int_alias(default: int, *names: str) -> int:
    raw = _env_first_non_empty(*names)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _to_float_alias(default: float, *names: str) -> float:
    raw = _env_first_non_empty(*names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _to_bool_alias(default: bool, *names: str) -> bool:
    raw = _env_first_non_empty(*names)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}



def _to_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default



def _to_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default



def _to_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _to_bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _sanitize_banner_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", (value or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or fallback


def _normalize_banner_type(value: Any) -> str:
    lowered = str(value or "info").strip().lower()
    if lowered in {"success", "warning", "error", "info"}:
        return lowered
    return "info"


def _clean_banner_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:limit]


def _decode_env_text(value: str) -> str:
    # Allow writing newline as \n in .env while keeping plain values readable.
    return value.replace("\\n", "\n")


def _parse_ui_banners() -> tuple[dict[str, Any], ...]:
    raw = (
        _env_first_non_empty("APP_UI_BANNERS", "UI_BANNERS", "WEBUI_BANNERS")
        or ""
    )
    if not raw:
        return ()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        message = _clean_banner_text(raw, 2000)
        if not message:
            return ()
        return (
            {
                "id": "banner-1",
                "type": "info",
                "title": "",
                "message": message,
                "dismissible": True,
            },
        )

    items: list[Any]
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = [parsed]
    else:
        return ()

    banners: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        fallback_id = f"banner-{index}"

        if isinstance(item, str):
            message = _clean_banner_text(item, 2000)
            if not message:
                continue
            banners.append(
                {
                    "id": fallback_id,
                    "type": "info",
                    "title": "",
                    "message": message,
                    "dismissible": True,
                }
            )
            continue

        if not isinstance(item, dict):
            continue
        if not _to_bool_value(item.get("enabled"), True):
            continue

        title = _clean_banner_text(item.get("title"), 200)
        message = _clean_banner_text(
            item.get("message") or item.get("content") or item.get("text") or item.get("body"),
            2000,
        )
        if not message:
            continue

        banners.append(
            {
                "id": _sanitize_banner_id(str(item.get("id") or ""), fallback_id),
                "type": _normalize_banner_type(item.get("type") or item.get("level")),
                "title": title,
                "message": message,
                "dismissible": _to_bool_value(item.get("dismissible"), True),
            }
        )

    return tuple(banners)


DEFAULT_SOC_PROMPT_TEMPLATE = (
    "SoC, ASIC, chiplet, CPU, GPU, NPU, DSP, ISP, VPU, DPU, MCU, PMU, NoC, interconnect, "
    "AXI, AXI4, AXI-Lite, AHB, APB, ACE, CHI, UCIe, PCIe, CXL, DDR, DDR4, DDR5, LPDDR4, "
    "LPDDR5, HBM, SRAM, ROM, eMMC, UFS, PHY, SerDes, PLL, DLL, RC oscillator, clock, "
    "clock tree, clock gating, reset, async reset, sync reset, power domain, voltage island, "
    "retention, isolation, level shifter, DVFS, AVS, UPF, CPF, RTL, SystemVerilog, Verilog, "
    "VHDL, UVM, testbench, assertion, SVA, lint, SpyGlass, CDC, RDC, STA, MCMM, OCV, AOCV, "
    "POCV, derate, setup, hold, recovery, removal, skew, jitter, uncertainty, timing closure, "
    "timing path, false path, multicycle path, path group, endpoint, startpoint, slack, WNS, "
    "TNS, violating path, critical path, synthesis, logic synthesis, Design Compiler, Genus, "
    "netlist, mapped netlist, unmapped netlist, compile, incremental compile, retiming, "
    "boundary optimization, datapath optimization, resource sharing, register balancing, ECO, "
    "formal, equivalence check, LEC, Conformal, Formality, gate-level simulation, GLS, SDF, "
    "back annotation, place and route, place-and-route, PnR, floorplan, floorplanning, macro "
    "placement, standard cell, utilization, density, congestion, global placement, detailed "
    "placement, legalization, CTS, clock tree synthesis, useful skew, hold fixing, setup fixing, "
    "routing, global route, detailed route, track assignment, antenna, filler cell, decap, "
    "tap cell, endcap, spare cell, spare gate, metal fill, density fill, ECO route, route guide, "
    "signoff, sign-off, DRC, LVS, ERC, extraction, parasitic extraction, RC extraction, SPEF, "
    "DEF, LEF, Liberty, .lib, TLU+, QRC, StarRC, Quantus, IR drop, dynamic IR drop, static IR drop, "
    "EM, electromigration, voltage drop, power integrity, signal integrity, SI, crosstalk, noise, "
    "glitch, overshoot, undershoot, hotspot, thermal, leakage, dynamic power, switching power, "
    "internal power, leakage power, power analysis, PrimeTime PX, PrimePower, Voltus, RedHawk, "
    "vectorless, VCD, FSDB, SAIF, toggle rate, activity factor, inrush current, rush current, "
    "decoupling capacitor, decap cell, package model, bump, substrate, interposer, TSV, process "
    "node, 28nm, 16nm, 12nm, 7nm, 5nm, 4nm, 3nm, FinFET, GAA, foundry, TSMC, Samsung, Intel, "
    "PDK, DFM, manufacturability, yield, wafer, lot, mask, reticle, tape-out, respin, metal fix, "
    "MPW, shuttle, bring-up, validation, characterization, errata, workaround, DFT, scan, scan chain, "
    "scan compression, EDT, ATPG, stuck-at, transition fault, path delay fault, bridging fault, "
    "JTAG, boundary scan, MBIST, LBIST, BISR, repair, fuse, eFuse, OTP, secure boot, TrustZone, "
    "TEE, firmware, bootloader, NAND, NAND flash, Toggle NAND, ONFI, raw NAND, managed NAND, SLC, "
    "MLC, TLC, QLC, PLC, 3D NAND, V-NAND, charge trap, floating gate, page, block, plane, die, LUN, "
    "bad block, bad block management, BBT, ECC, BCH, LDPC, RAID, read disturb, program disturb, "
    "erase disturb, wear leveling, garbage collection, overprovisioning, endurance, retention, BER, "
    "bit error rate, read retry, soft decoding, threshold voltage, ISPP, incremental step pulse "
    "programming, erase verify, program verify, copyback, cache read, cache program, multi-plane, "
    "interleaving, channel, CE, RE, WE, ALE, CLE, R/B, spare area, OOB, metadata, FTL, flash "
    "translation layer, NVMe, SATA, controller, queue depth, throughput, latency, bandwidth, QoS, "
    "arbiter, scheduler, mux, demux, crossbar, SRAM compiler, memory compiler, register file, dual "
    "port RAM, single port RAM, SRAM macro, macro, hard macro, soft macro, black box, hierarchy, "
    "partition, block-level, top-level, full-chip, chip top, top module, hierarchy flattening, "
    "dont_touch, set_false_path, set_multicycle_path, create_clock, generated clock, propagated "
    "clock, ideal clock, set_input_delay, set_output_delay, set_clock_uncertainty, set_clock_groups, "
    "operating condition, corner, slow corner, fast corner, typical corner, SS, FF, TT, RCmax, "
    "RCmin, setup view, hold view."
)


def _parse_prompt_templates() -> tuple[dict[str, str], ...]:
    raw = _env_first_non_empty("APP_PROMPT_TEMPLATES", "APP_SOC_PROMPT_TEMPLATE") or ""
    if not raw:
        return (
            {
                "id": "soc-design",
                "label": "SoC設計テンプレート",
                "content": DEFAULT_SOC_PROMPT_TEMPLATE,
            },
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        content = _decode_env_text(raw).strip()
        if not content:
            return ()
        return (
            {
                "id": "template-1",
                "label": "Template 1",
                "content": content,
            },
        )

    items: list[Any]
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, str):
        items = [parsed]
    else:
        return ()

    templates: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            content = _decode_env_text(item).strip()
            if not content:
                continue
            templates.append(
                {
                    "id": f"template-{index}",
                    "label": f"Template {index}",
                    "content": content,
                }
            )
            continue

        if not isinstance(item, dict):
            continue
        content = _decode_env_text(str(item.get("content") or item.get("prompt") or "")).strip()
        if not content:
            continue
        label = str(item.get("label") or item.get("name") or f"Template {index}").strip() or f"Template {index}"
        template_id = _sanitize_banner_id(str(item.get("id") or ""), f"template-{index}")
        templates.append(
            {
                "id": template_id,
                "label": label[:40],
                "content": content,
            }
        )

    return tuple(templates)



def load_settings() -> Settings:
    app_data_dir_raw = _env_first_non_empty("APP_DATA_DIR", "DATA_DIR")
    if app_data_dir_raw:
        app_data_dir = Path(app_data_dir_raw)
    else:
        app_data_dir = Path("data")

    transcripts_dir = Path(
        _env_first_non_empty("APP_TRANSCRIPTS_DIR")
        or str(app_data_dir / "transcripts")
    )
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    diarization_work_dir = Path(
        _env_first_non_empty("DIARIZATION_WORK_DIR")
        or str(app_data_dir / "diarization")
    )
    diarization_work_dir.mkdir(parents=True, exist_ok=True)

    asr_api_key = _env_first_non_empty("ASR_API_KEY", "OPENAI_API_KEY") or ""
    base_url = _env_first_non_empty("ASR_BASE_URL", "OPENAI_BASE_URL")
    summary_api_key = _env_first_non_empty("SUMMARY_API_KEY") or asr_api_key
    summary_base_url = _env_first_non_empty("SUMMARY_BASE_URL") or base_url
    summary_system_prompt = _decode_env_text(_env_first_non_empty("SUMMARY_SYSTEM_PROMPT") or "")
    summary_prompt_template = _decode_env_text(_env_first_non_empty("SUMMARY_PROMPT_TEMPLATE") or "")
    proofread_api_key = (
        _env_first_non_empty("PROOFREAD_API_KEY")
        or summary_api_key
        or asr_api_key
    )
    proofread_base_url = _env_first_non_empty("PROOFREAD_BASE_URL") or summary_base_url or base_url
    proofread_system_prompt = _decode_env_text(_env_first_non_empty("PROOFREAD_SYSTEM_PROMPT") or "")
    proofread_prompt_template = _decode_env_text(_env_first_non_empty("PROOFREAD_PROMPT_TEMPLATE") or "")

    asr_model = (
        _env_first_non_empty("ASR_MODEL")
        or _env_first_non_empty("WHISPER_MODEL")
        or "mistralai/Voxtral-Mini-4B-Realtime-2602"
    )
    app_brand_title = _decode_env_text(_env_first_non_empty("APP_BRAND_TITLE") or "whistx")
    app_brand_tagline = _decode_env_text(
        _env_first_non_empty("APP_BRAND_TAGLINE")
        or "高精度リアルタイム文字起こし"
    )
    return Settings(
        host=_env_first_non_empty("APP_HOST", "HOST") or "0.0.0.0",
        port=_to_int_alias(8005, "APP_PORT", "PORT"),
        ws_path=_env_first_non_empty("APP_WS_PATH", "WS_PATH") or "/ws/transcribe",
        transcripts_dir=transcripts_dir,
        openai_api_key=asr_api_key,
        openai_base_url=base_url,
        asr_model=asr_model,
        summary_api_key=summary_api_key,
        summary_base_url=summary_base_url,
        summary_model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        summary_temperature=_to_float("SUMMARY_TEMPERATURE", 0.2),
        summary_input_max_chars=max(2_000, _to_int("SUMMARY_INPUT_MAX_CHARS", 16_000)),
        summary_system_prompt=summary_system_prompt,
        summary_prompt_template=summary_prompt_template,
        proofread_api_key=proofread_api_key,
        proofread_base_url=proofread_base_url,
        proofread_model=os.getenv("PROOFREAD_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        proofread_temperature=_to_float("PROOFREAD_TEMPERATURE", 0.0),
        proofread_input_max_chars=max(2_000, _to_int("PROOFREAD_INPUT_MAX_CHARS", 24_000)),
        proofread_system_prompt=proofread_system_prompt,
        proofread_prompt_template=proofread_prompt_template,
        diarization_enabled=_to_bool("DIARIZATION_ENABLED", False),
        diarization_hf_token=os.getenv("DIARIZATION_HF_TOKEN", "").strip(),
        diarization_model=(
            os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1").strip()
            or "pyannote/speaker-diarization-3.1"
        ),
        diarization_device=os.getenv("DIARIZATION_DEVICE", "auto").strip() or "auto",
        diarization_sample_rate=max(8_000, _to_int("DIARIZATION_SAMPLE_RATE", 16_000)),
        diarization_num_speakers=max(0, _to_int("DIARIZATION_NUM_SPEAKERS", 0)),
        diarization_min_speakers=max(0, _to_int("DIARIZATION_MIN_SPEAKERS", 0)),
        diarization_max_speakers=max(0, _to_int("DIARIZATION_MAX_SPEAKERS", 0)),
        diarization_work_dir=diarization_work_dir,
        diarization_keep_chunks=_to_bool("DIARIZATION_KEEP_CHUNKS", False),
        ffmpeg_bin=(
            _env_first_non_empty("DIARIZATION_FFMPEG_BIN", "FFMPEG_BIN")
            or "ffmpeg"
        ),
        asr_preprocess_enabled=_to_bool_alias(True, "ASR_PREPROCESS_ENABLED"),
        asr_preprocess_sample_rate=max(8_000, _to_int_alias(16_000, "ASR_PREPROCESS_SAMPLE_RATE")),
        asr_overlap_ms=max(0, _to_int_alias(3_500, "ASR_OVERLAP_MS")),
        default_language=_env_first_non_empty("ASR_DEFAULT_LANGUAGE", "DEFAULT_LANGUAGE") or "ja",
        default_prompt=_env_first_non_empty("ASR_DEFAULT_PROMPT", "DEFAULT_PROMPT") or "",
        default_temperature=_to_float_alias(0.0, "ASR_DEFAULT_TEMPERATURE", "DEFAULT_TEMPERATURE"),
        context_prompt_enabled=_to_bool_alias(True, "ASR_CONTEXT_PROMPT_ENABLED", "CONTEXT_PROMPT_ENABLED"),
        context_max_chars=max(0, _to_int_alias(1000, "ASR_CONTEXT_MAX_CHARS", "CONTEXT_MAX_CHARS")),
        context_recent_lines=max(1, _to_int_alias(2, "ASR_CONTEXT_RECENT_LINES")),
        context_term_limit=max(8, _to_int_alias(48, "ASR_CONTEXT_TERM_LIMIT")),
        max_queue_size=max(1, _to_int_alias(8, "ASR_MAX_QUEUE_SIZE", "MAX_QUEUE_SIZE")),
        max_chunk_bytes=max(
            1024,
            _to_int_alias(12 * 1024 * 1024, "ASR_MAX_CHUNK_BYTES", "MAX_CHUNK_BYTES"),
        ),
        app_brand_title=app_brand_title,
        app_brand_tagline=app_brand_tagline,
        ui_prompt_templates=_parse_prompt_templates(),
        ui_banners=_parse_ui_banners(),
    )


settings = load_settings()

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple


ORACLE_FIELDS = {
    "frame",
    "timestamp",
    "chunk",
    "phase_start",
    "phase_end",
    "gold_phase",
    "gold_event",
    "gold_fsm",
}

ANCHOR_POLICIES = {"static", "dynamic_affordance"}

EFFECT_TYPES = {
    "grasp_change",
    "release_change",
    "placement_change",
    "containment_change",
    "opening_change",
    "closing_change",
    "shape_change",
    "topology_change",
    "alignment_change",
    "surface_change",
    "classification_change",
    "unknown",
}

DYNAMIC_AFFORDANCE_ANCHORS = {
    "corner",
    "edge",
    "crease",
    "fold_region",
    "contact_patch",
}

PASSTHROUGH_FIELDS = (
    "dataset_name",
    "dataset_uuid",
    "dataset_batch_number",
    "env_type",
    "scene_level1",
    "scene_level2",
    "atomic_actions",
    "operation_platform_height",
    "device_model",
    "end_effector_type",
    "task_operation_type",
    "tele_type",
    "dataset_name_id",
    "yaml_file_path",
    "data_path",
)


def normalize_object_id(name: str) -> str:
    raw = str(name or "").strip().lower()
    normalized = re.sub(r"[\s-]+", "_", raw)
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_") or "object"


def normalize_reference(reference: Any) -> str:
    text = str(reference or "").strip()
    if not text:
        return ""
    if ":" in text:
        object_id, anchor = text.split(":", 1)
    else:
        object_id, anchor = text, "main_body"
    return f"{normalize_object_id(object_id)}:{normalize_object_id(anchor) or 'main_body'}"


def _strip_oracle_fields(value: Any, path: str = "") -> Tuple[Any, List[str]]:
    errors = []
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in ORACLE_FIELDS:
                errors.append(f"Removed oracle field '{path + key}'.")
                continue
            cleaned_item, nested_errors = _strip_oracle_fields(item, f"{path}{key}.")
            cleaned[key] = cleaned_item
            errors.extend(nested_errors)
        return cleaned, errors
    if isinstance(value, list):
        cleaned_list = []
        for idx, item in enumerate(value):
            cleaned_item, nested_errors = _strip_oracle_fields(item, f"{path}{idx}.")
            cleaned_list.append(cleaned_item)
            errors.extend(nested_errors)
        return cleaned_list, errors
    return value, errors


def _as_instruction(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if len(value) == 1 else value
    return value or ""


def _input_objects(raw_objects: List[Any]) -> List[Dict[str, Any]]:
    objects = []
    for raw in raw_objects or []:
        if isinstance(raw, str):
            objects.append({"name": raw, "color": "", "anchor_policy": ""})
            continue
        if not isinstance(raw, dict):
            continue
        obj = dict(raw)
        if "name" not in obj and "object_name" in obj:
            obj["name"] = obj["object_name"]
        objects.append(obj)
    return objects


def _export_objects(raw_objects: List[Any]) -> List[Dict[str, Any]]:
    objects = _input_objects(raw_objects)
    generated_ids = [normalize_object_id(obj.get("id") or obj.get("name")) for obj in objects]
    duplicated_ids = {object_id for object_id in generated_ids if generated_ids.count(object_id) > 1}
    seen: Dict[str, int] = {}
    exported = []

    for obj, generated_id in zip(objects, generated_ids):
        seen[generated_id] = seen.get(generated_id, 0) + 1
        resolved_id = f"{generated_id}_{seen[generated_id]}" if generated_id in duplicated_ids else generated_id
        output = {
            "id": resolved_id,
            "name": str(obj.get("name") or "").strip(),
            "color": str(obj.get("color") or "").strip(),
            "anchor_policy": str(obj.get("anchor_policy") or "").strip(),
        }
        exported.append(output)
    return exported


def _object_ids(objects: List[Dict[str, Any]]) -> List[str]:
    return [normalize_object_id(obj.get("id") or obj.get("name")) for obj in objects]


def _object_lookup(objects: List[Dict[str, Any]]) -> Dict[str, str]:
    lookup = {}
    for obj in objects:
        object_id = normalize_object_id(obj.get("id") or obj.get("name"))
        lookup[object_id] = object_id
        lookup[normalize_object_id(obj.get("name"))] = object_id
    return lookup


def _normalize_registered_reference(reference: Any, lookup: Dict[str, str]) -> str:
    normalized = normalize_reference(reference)
    if not normalized:
        return ""
    object_id, anchor = normalized.split(":", 1)
    return f"{lookup.get(object_id, object_id)}:{anchor}"


def _build_target_sequence(data: Dict[str, Any], lookup: Dict[str, str]) -> List[str]:
    sequence = []
    for raw in data.get("target_sequence") or []:
        reference = raw
        if isinstance(raw, dict):
            reference = raw.get("canonical_id") or raw.get("id") or raw.get("object_name")
        normalized = _normalize_registered_reference(reference, lookup)
        if normalized:
            sequence.append(normalized)
    return sequence


def _build_context(data: Dict[str, Any], lookup: Dict[str, str]) -> Dict[str, str]:
    context = {}
    for key, value in (data.get("context") or {}).items():
        if key and value:
            context[key] = _normalize_registered_reference(value, lookup)
    return context


def _build_expected_effects(data: Dict[str, Any], lookup: Dict[str, str]) -> List[Dict[str, Any]]:
    effects = []
    for raw in data.get("expected_effects") or []:
        if not isinstance(raw, dict):
            continue
        effect = {}
        if raw.get("object") or raw.get("object_name"):
            effect["object"] = _normalize_registered_reference(
                raw.get("object") or raw.get("object_name"),
                lookup,
            )
        effect_types = raw.get("effect_type")
        if effect_types:
            effect["effect_type"] = effect_types
        for key in ("from_state", "to_state"):
            if raw.get(key) not in ("", None):
                effect[key] = raw[key]
        if effect:
            effects.append(effect)
    return effects


def validate_preannotation(data: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    warnings = []
    objects = _input_objects(data.get("objects") or [])
    object_ids = _object_ids(objects)
    object_by_id = {object_id: obj for object_id, obj in zip(object_ids, objects)}

    if not data.get("task_type"):
        errors.append("task_type is required.")
    if not data.get("task_instruction"):
        errors.append("task_instruction is required.")
    if not objects:
        errors.append("objects must contain at least one object.")

    if len(object_ids) != len(set(object_ids)):
        errors.append("objects contain duplicate ids; assign an explicit id to each same-name object.")

    for index, obj in enumerate(objects):
        prefix = f"objects[{index}]"
        if not obj.get("name"):
            errors.append(f"{prefix}.name is required.")
        if not obj.get("color"):
            errors.append(f"{prefix}.color is required.")
        if obj.get("anchor_policy") not in ANCHOR_POLICIES:
            errors.append(f"{prefix}.anchor_policy must be static or dynamic_affordance.")

    def validate_reference(reference: Any, path: str) -> Tuple[str, str] | None:
        normalized = normalize_reference(reference)
        if not normalized:
            errors.append(f"{path} is required.")
            return None
        object_id, anchor = normalized.split(":", 1)
        if object_id not in object_by_id:
            errors.append(f"{path} references unknown object '{object_id}'.")
            return None
        return object_id, anchor

    target_sequence = data.get("target_sequence") or []
    for index, reference in enumerate(target_sequence):
        parsed = validate_reference(reference, f"target_sequence[{index}]")
        if not parsed:
            continue
        object_id, anchor = parsed
        obj = object_by_id[object_id]
        if obj.get("anchor_policy") != "static":
            errors.append(f"target_sequence[{index}] references '{object_id}', which is not static.")
        if anchor in DYNAMIC_AFFORDANCE_ANCHORS:
            errors.append(f"target_sequence[{index}] uses dynamic affordance anchor '{anchor}'.")

    for key, reference in (data.get("context") or {}).items():
        validate_reference(reference, f"context.{key}")

    effects = data.get("expected_effects") or []
    if not effects:
        errors.append("expected_effects must contain at least one task-level effect.")
    for index, effect in enumerate(effects):
        if not isinstance(effect, dict):
            errors.append(f"expected_effects[{index}] must be an object.")
            continue
        validate_reference(effect.get("object"), f"expected_effects[{index}].object")
        effect_types = effect.get("effect_type")
        if isinstance(effect_types, str):
            effect_types = [effect_types]
        if not effect_types or any(effect_type not in EFFECT_TYPES for effect_type in effect_types):
            errors.append(f"expected_effects[{index}].effect_type is not supported.")

    return {
        "status": "error" if errors else "warning" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
    }


def build_preannotation_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_data, _ = _strip_oracle_fields(deepcopy(data))
    objects = _export_objects(cleaned_data.get("objects") or [])
    lookup = _object_lookup(objects)

    result = {
        "task_type": cleaned_data.get("task_type") or "",
        "task_instruction": _as_instruction(cleaned_data.get("task_instruction")),
        "objects": objects,
    }

    target_sequence = _build_target_sequence(cleaned_data, lookup)
    if target_sequence:
        result["target_sequence"] = target_sequence

    context = _build_context(cleaned_data, lookup)
    if context:
        result["context"] = context

    result["expected_effects"] = _build_expected_effects(cleaned_data, lookup)

    for field in PASSTHROUGH_FIELDS:
        if field in cleaned_data:
            result[field] = cleaned_data[field]

    return result

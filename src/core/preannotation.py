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

TARGET_SEQUENCE_ROLES = {
    "manipulated_object",
    "target",
    "labware",
    "deformable_object",
}

NON_TARGET_ROLES = {
    "container",
    "support_surface",
    "destination_region",
    "fixture",
}

DEFORMABLE_KEYWORDS = {
    "cloth",
    "towel",
    "garment",
    "shirt",
    "t-shirt",
    "hoodie",
    "pants",
    "trousers",
    "bag",
    "trash bag",
    "quilt",
    "bed sheet",
}

OBJECT_ALIASES = {
    "包子": "baozi",
    "馒头": "steamed_bun",
    "蒸笼": "steamer",
    "蒸笼盖": "steamer_lid",
    "桌面": "table",
    "桌子": "table",
}


def normalize_object_id(name: str) -> str:
    raw = str(name or "").strip()
    if raw in OBJECT_ALIASES:
        return OBJECT_ALIASES[raw]
    lowered = raw.lower()
    lowered = lowered.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "object"


def default_anchor_for_role(role: str) -> str:
    role = str(role or "").strip()
    if role == "support_surface":
        return "main_surface"
    if role == "destination_region":
        return "inner_zone"
    return "main_body"


def _split_canonical_id(canonical_id: str) -> Tuple[str, str]:
    text = str(canonical_id or "").strip()
    if ":" in text:
        object_id, anchor_id = text.split(":", 1)
        return object_id.strip(), anchor_id.strip() or "main_body"
    return text, "main_body"


def _destination_object_id(object_name: str, role: str) -> Tuple[str, str]:
    object_id = normalize_object_id(object_name)
    anchor = default_anchor_for_role(role)
    if role == "destination_region" and object_id.endswith("_inner_zone"):
        return object_id[: -len("_inner_zone")] or "object", "inner_zone"
    return object_id, anchor


def build_unique_canonical_ids(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = [dict(obj) for obj in objects if obj.get("object_name")]
    base_ids = []
    for obj in enriched:
        role = obj.get("role") or "manipulated_object"
        base_id, anchor = _destination_object_id(obj.get("object_name", ""), role)
        obj["_base_object_id"] = base_id
        obj["_default_anchor"] = anchor
        obj["role"] = role
        base_ids.append(base_id)

    counts = {base_id: base_ids.count(base_id) for base_id in set(base_ids)}
    seen = {}
    for obj in enriched:
        base_id = obj["_base_object_id"]
        seen[base_id] = seen.get(base_id, 0) + 1
        object_id = f"{base_id}_{seen[base_id]}" if counts[base_id] > 1 else base_id
        anchor = obj.get("anchor") or obj["_default_anchor"]
        obj["canonical_id"] = f"{object_id}:{anchor}"
        obj.pop("_base_object_id", None)
        obj.pop("_default_anchor", None)
    return enriched


def _object_lookup(objects: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for obj in objects:
        canonical_id = obj.get("canonical_id")
        if canonical_id:
            lookup[canonical_id] = obj
        object_name = obj.get("object_name")
        if object_name:
            lookup[str(object_name)] = obj
            lookup[normalize_object_id(object_name)] = obj
    return lookup


def _target_order_value(obj: Dict[str, Any]) -> int | None:
    value = obj.get("target_order")
    if value in ("", None):
        return None
    try:
        order = int(value)
    except (TypeError, ValueError):
        return None
    return order if order > 0 else None


def _target_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "canonical_id": obj["canonical_id"],
        "object_name": obj["object_name"],
        "role": obj.get("role", "manipulated_object"),
    }


def build_target_sequence(
    objects: List[Dict[str, Any]],
    requested_sequence: List[Dict[str, Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings = []
    lookup = _object_lookup(objects)

    ordered = [(obj, _target_order_value(obj)) for obj in objects]
    ordered = [(obj, order) for obj, order in ordered if order is not None]
    if ordered:
        sequence_objects = [obj for obj, _ in sorted(ordered, key=lambda item: item[1])]
    elif requested_sequence:
        sequence_objects = []
        for row in requested_sequence:
            key = row.get("canonical_id") or row.get("object_name")
            object_id, _ = _split_canonical_id(key)
            obj = lookup.get(key) or lookup.get(object_id)
            if obj:
                sequence_objects.append(obj)
    else:
        sequence_objects = [
            obj for obj in objects if obj.get("role", "manipulated_object") in TARGET_SEQUENCE_ROLES
        ]

    sequence = []
    seen = set()
    for obj in sequence_objects:
        canonical_id = obj.get("canonical_id")
        role = obj.get("role", "manipulated_object")
        if not canonical_id or canonical_id in seen:
            continue
        if role in NON_TARGET_ROLES:
            warnings.append(
                f"{canonical_id} has role '{role}' and should not be in target_sequence unless it is actively manipulated."
            )
        sequence.append(_target_entry(obj))
        seen.add(canonical_id)
    return sequence, warnings


def _first_by_role(objects: List[Dict[str, Any]], roles: set[str]) -> Dict[str, Any] | None:
    for obj in objects:
        if obj.get("role") in roles:
            return obj
    return None


def build_context(data: Dict[str, Any], objects: List[Dict[str, Any]]) -> Dict[str, str]:
    context = dict(data.get("context") or {})
    lookup = _object_lookup(objects)
    container = _first_by_role(objects, {"container"})
    support_surface = _first_by_role(objects, {"support_surface"})
    destination_region = _first_by_role(objects, {"destination_region"})

    if not context.get("container") and container:
        context["container"] = container["canonical_id"]
    if not context.get("support_surface") and support_surface:
        context["support_surface"] = support_surface["canonical_id"]
    if not context.get("destination"):
        if destination_region:
            context["destination"] = destination_region["canonical_id"]
        elif container:
            container_id, _ = _split_canonical_id(container["canonical_id"])
            context["destination"] = f"{container_id}:inner_zone"

    for flat_key, nested_key in (
        ("context_destination", "destination"),
        ("context_container", "container"),
        ("context_support_surface", "support_surface"),
    ):
        if data.get(flat_key):
            context[nested_key] = data[flat_key]

    for nested_key, value in list(context.items()):
        obj = lookup.get(value) or lookup.get(normalize_object_id(value))
        if obj and nested_key == "destination" and obj.get("role") == "container":
            container_id, _ = _split_canonical_id(obj["canonical_id"])
            context[nested_key] = f"{container_id}:inner_zone"
        elif obj:
            context[nested_key] = obj["canonical_id"]

    return {key: value for key, value in context.items() if value}


def _contains_container_placement_language(instructions: List[str]) -> bool:
    text = " ".join(instructions).lower()
    return any(token in text for token in ("放入", "put into", "place into", "insert into", "into the"))


def _state_suffix(canonical_id: str) -> str:
    object_id, _ = _split_canonical_id(canonical_id)
    return object_id


def build_expected_effects(
    data: Dict[str, Any],
    objects: List[Dict[str, Any]],
    target_sequence: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings = []
    raw_effects = [dict(effect) for effect in data.get("expected_effects") or [] if effect]
    lookup = _object_lookup(objects)
    effects = []

    for effect in raw_effects:
        obj_ref = effect.get("object") or effect.get("object_name")
        obj = lookup.get(obj_ref) or lookup.get(normalize_object_id(obj_ref))
        if obj:
            effect["object"] = obj["canonical_id"]
        effect.pop("object_name", None)
        if effect.get("object") and effect.get("effect_type"):
            effects.append(effect)

    instructions = data.get("task_instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]

    needs_containment = (
        data.get("task_type") == "container_placement"
        or _contains_container_placement_language(instructions)
    )
    has_containment = any(effect.get("effect_type") == "containment_change" for effect in effects)
    if needs_containment and not has_containment:
        manipulated = _first_by_role(objects, {"manipulated_object", "target", "deformable_object"})
        container = _first_by_role(objects, {"container"})
        if manipulated and container:
            container_state = _state_suffix(container["canonical_id"])
            effects.append(
                {
                    "object": manipulated["canonical_id"],
                    "effect_type": "containment_change",
                    "from_state": f"outside_{container_state}",
                    "to_state": f"inside_{container_state}",
                }
            )
        else:
            warnings.append("Task looks like container placement but lacks a manipulated object or container.")

    return effects, warnings


def _strip_oracle_fields(value: Any, path: str = "") -> Tuple[Any, List[str]]:
    warnings = []
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in ORACLE_FIELDS:
                warnings.append(f"Removed oracle field '{path + key}'.")
                continue
            cleaned_item, nested_warnings = _strip_oracle_fields(item, f"{path}{key}.")
            cleaned[key] = cleaned_item
            warnings.extend(nested_warnings)
        return cleaned, warnings
    if isinstance(value, list):
        cleaned_list = []
        for idx, item in enumerate(value):
            cleaned_item, nested_warnings = _strip_oracle_fields(item, f"{path}{idx}.")
            cleaned_list.append(cleaned_item)
            warnings.extend(nested_warnings)
        return cleaned_list, warnings
    return value, warnings


def validate_preannotation(data: Dict[str, Any]) -> Dict[str, Any]:
    warnings = []
    assumptions = []

    object_ids = [obj.get("canonical_id") for obj in data.get("objects", []) if obj.get("canonical_id")]
    target_ids = [
        obj.get("canonical_id") for obj in data.get("target_sequence", []) if obj.get("canonical_id")
    ]

    if len(object_ids) != len(set(object_ids)):
        warnings.append("objects contain duplicate canonical_id values.")
    if len(target_ids) != len(set(target_ids)):
        warnings.append("target_sequence contains duplicate canonical_id values.")

    object_set = set(object_ids)
    target_set = set(target_ids)
    role_by_id = {obj.get("canonical_id"): obj.get("role") for obj in data.get("objects", [])}

    for canonical_id in target_ids:
        if canonical_id not in object_set:
            warnings.append(f"target_sequence canonical_id '{canonical_id}' is not present in objects.")
        role = role_by_id.get(canonical_id)
        if role in NON_TARGET_ROLES:
            warnings.append(f"{canonical_id} role '{role}' should not be in target_sequence.")

    context = data.get("context") or {}
    destination = context.get("destination")
    if destination in target_set and role_by_id.get(destination) != "manipulated_object":
        warnings.append("context.destination should not appear in target_sequence.")

    for effect in data.get("expected_effects", []) or []:
        effect_object = effect.get("object")
        if effect_object and effect_object not in object_set and effect_object not in target_set:
            warnings.append(f"expected_effects object '{effect_object}' is not present in objects.")

    if data.get("task_type") == "container_placement" and not (
        context.get("container") or context.get("destination")
    ):
        warnings.append("container_placement requires context.container or context.destination.")

    instructions = data.get("task_instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    if _contains_container_placement_language(instructions):
        if not any(effect.get("effect_type") == "containment_change" for effect in data.get("expected_effects", [])):
            warnings.append("Instruction indicates placement into a container but no containment_change effect exists.")

    if data.get("task_type") != "garment_folding" and data.get("deformable_anchor_policy"):
        warnings.append("deformable_anchor_policy should only be used for deformable/garment tasks.")

    if data.get("task_type") == "container_placement":
        assumptions.append("Container placement destination is represented in context, not target_sequence.")

    return {
        "status": "warning" if warnings else "pass",
        "warnings": warnings,
        "assumptions": assumptions,
    }


def build_preannotation_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_data, oracle_warnings = _strip_oracle_fields(deepcopy(data))
    objects = build_unique_canonical_ids(cleaned_data.get("objects") or [])
    task_type_warnings = []
    instructions = cleaned_data.get("task_instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    has_container = any(obj.get("role") == "container" for obj in objects)
    if not cleaned_data.get("task_type") and has_container and _contains_container_placement_language(instructions):
        cleaned_data["task_type"] = "container_placement"
    elif (
        cleaned_data.get("task_type") == "pick_and_place"
        and has_container
        and _contains_container_placement_language(instructions)
    ):
        task_type_warnings.append(
            "Instruction is container-style pick/place; consider task_type 'container_placement' if downstream supports it."
        )
    target_sequence, target_warnings = build_target_sequence(
        objects,
        cleaned_data.get("target_sequence") or None,
    )
    context = build_context(cleaned_data, objects)
    expected_effects, effect_warnings = build_expected_effects(cleaned_data, objects, target_sequence)

    result = dict(cleaned_data)
    for key in ("context_destination", "context_container", "context_support_surface"):
        result.pop(key, None)
    object_export_keys = ("canonical_id", "object_name", "role", "color")
    exported_objects = []
    for obj in objects:
        exported = {
            key: obj[key]
            for key in object_export_keys
            if key in obj and obj[key] not in ("", None)
        }
        for key, value in obj.items():
            if key not in exported and key not in ("target_order", "anchor") and value not in ("", None):
                exported[key] = value
        exported_objects.append(exported)
    result["objects"] = exported_objects
    result["target_sequence"] = target_sequence
    if context:
        result["context"] = context
    if expected_effects:
        result["expected_effects"] = expected_effects

    if result.get("task_type") == "garment_folding":
        if any(obj.get("role") == "deformable_object" for obj in objects):
            result["deformable_anchor_policy"] = "dynamic_affordance"
    else:
        result.pop("deformable_anchor_policy", None)

    validation = validate_preannotation(result)
    warnings = []
    warnings.extend(oracle_warnings)
    warnings.extend(task_type_warnings)
    warnings.extend(target_warnings)
    warnings.extend(effect_warnings)
    warnings.extend(validation["warnings"])
    if warnings:
        result["warnings"] = list(dict.fromkeys(warnings))
        result["needs_review"] = True
    else:
        result.pop("warnings", None)
        result["needs_review"] = False

    validation["warnings"] = result.get("warnings", [])
    validation["status"] = "warning" if validation["warnings"] else validation["status"]
    result["validation"] = validation
    return result

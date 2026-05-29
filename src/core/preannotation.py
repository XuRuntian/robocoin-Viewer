import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple


SCHEMA_VERSION = "task_prior_v1_5"

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

ACTIVE_ROLES = {
    "manipulated_object",
    "target",
    "labware",
    "deformable_object",
}

# Backward-compatible name used by the Streamlit UI.
TARGET_SEQUENCE_ROLES = ACTIVE_ROLES

NON_TARGET_ROLES = {
    "container",
    "support_surface",
    "destination",
    "destination_region",
    "fixture",
}

OVER_SPECIFIC_ROLES = {
    "container",
    "support_surface",
    "labware",
    "deformable_object",
}

CONTAINER_NAMES = {
    "basket",
    "box",
    "bowl",
    "steamer",
    "double-ear steamer",
    "lunch box",
    "open top box",
    "tray",
    "蒸笼",
    "篮子",
}

SUPPORT_SURFACE_NAMES = {
    "table",
    "counter",
    "desk",
    "桌子",
    "桌面",
    "台面",
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
    "毛巾",
    "衣服",
    "垃圾袋",
}

SOFT_RIGID_NAMES = {
    "baozi",
    "steamed bun",
    "包子",
    "馒头",
}

DYNAMIC_AFFORDANCE_ANCHORS = {
    "corner",
    "edge",
    "crease",
    "fold_region",
    "contact_patch",
}

DEFAULT_DYNAMIC_AFFORDANCES = [
    "corner",
    "edge",
    "crease",
    "fold_region",
    "contact_patch",
]

DEFAULT_CONTAINER_ANCHORS = [
    "main_body",
    "inner_zone",
    "rim",
]

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


def _as_list(value: Any) -> List[Any]:
    if value in ("", None):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _contains_any(text: str, tokens: set[str]) -> bool:
    lowered = str(text or "").lower()
    return any(token.lower() in lowered for token in tokens)


def _contains_container_placement_language(instructions: List[str]) -> bool:
    text = " ".join(str(item) for item in instructions).lower()
    return any(token in text for token in ("放入", "put into", "place into", "insert into", "into the"))


def _contains_folding_language(instructions: List[str]) -> bool:
    text = " ".join(str(item) for item in instructions).lower()
    return any(token in text for token in ("fold", "folding", "叠", "折叠"))


def _input_objects(raw_objects: List[Any]) -> List[Dict[str, Any]]:
    objects = []
    for obj in raw_objects or []:
        if isinstance(obj, str):
            objects.append({"object_name": obj})
        elif isinstance(obj, dict) and obj.get("object_name"):
            objects.append(dict(obj))
    return objects


def _infer_task_type(data: Dict[str, Any], objects: List[Dict[str, Any]]) -> str:
    if data.get("task_type"):
        return data["task_type"]
    instructions = data.get("task_instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    has_fold = _contains_folding_language(instructions)
    has_into = _contains_container_placement_language(instructions)
    has_container = any(_object_is_container(obj) for obj in objects)
    if has_fold and has_into:
        return "compound_task"
    if has_fold:
        return "garment_folding"
    if has_into or has_container:
        return "container_placement"
    return "pick_and_place"


def _object_is_container(obj: Dict[str, Any]) -> bool:
    role = obj.get("role")
    if role in {"container", "destination", "destination_region"}:
        return True
    return _contains_any(obj.get("object_name", ""), CONTAINER_NAMES)


def _object_is_support_surface(obj: Dict[str, Any]) -> bool:
    if obj.get("role") == "support_surface":
        return True
    return _contains_any(obj.get("object_name", ""), SUPPORT_SURFACE_NAMES)


def _object_is_deformable(obj: Dict[str, Any], task_type: str) -> bool:
    if obj.get("object_type") == "deformable" or obj.get("role") == "deformable_object":
        return True
    if task_type in {"garment_folding", "compound_task"} and _contains_any(obj.get("object_name", ""), DEFORMABLE_KEYWORDS):
        return True
    return _contains_any(obj.get("object_name", ""), DEFORMABLE_KEYWORDS)


def _object_is_soft_rigid(obj: Dict[str, Any]) -> bool:
    return _contains_any(obj.get("object_name", ""), SOFT_RIGID_NAMES)


def _infer_semantic_role(obj: Dict[str, Any], task_type: str, index: int) -> str:
    role = obj.get("role")
    if role:
        return role
    if _object_is_support_surface(obj):
        return "support_surface"
    if _object_is_container(obj):
        return "destination"
    if _object_is_deformable(obj, task_type):
        return "deformable_object"
    return "manipulated_object"


def _infer_object_type(obj: Dict[str, Any], semantic_role: str, task_type: str) -> str:
    if obj.get("object_type"):
        return obj["object_type"]
    if semantic_role == "deformable_object" or _object_is_deformable(obj, task_type):
        return "deformable"
    if _object_is_soft_rigid(obj):
        return "soft_rigid"
    if semantic_role in {"destination", "container", "destination_region", "support_surface", "fixture", "tool"}:
        return "rigid"
    return "unknown"


def _infer_anchor_policy(obj: Dict[str, Any], semantic_role: str, object_type: str) -> str:
    if obj.get("anchor_policy"):
        return obj["anchor_policy"]
    if object_type == "deformable":
        return "dynamic_affordance"
    if semantic_role in {"destination", "container", "destination_region", "support_surface", "fixture", "tool"}:
        return "stable_part"
    return "none"


def _canonical_base_and_anchor(obj: Dict[str, Any], semantic_role: str) -> Tuple[str, str]:
    object_id = normalize_object_id(obj.get("object_name", ""))
    anchor = obj.get("anchor") or default_anchor_for_role(semantic_role)
    if semantic_role == "destination_region" and object_id.endswith("_inner_zone"):
        object_id = object_id[: -len("_inner_zone")] or "object"
        anchor = "inner_zone"
    return object_id, anchor


def build_unique_canonical_ids(objects: List[Dict[str, Any]], task_type: str | None = None) -> List[Dict[str, Any]]:
    enriched = _input_objects(objects)
    task_type = task_type or "pick_and_place"
    base_ids = []

    for idx, obj in enumerate(enriched):
        semantic_role = _infer_semantic_role(obj, task_type, idx)
        object_type = _infer_object_type(obj, semantic_role, task_type)
        anchor_policy = _infer_anchor_policy(obj, semantic_role, object_type)
        base_id, anchor = _canonical_base_and_anchor(obj, semantic_role)

        obj["_semantic_role"] = semantic_role
        obj["_base_object_id"] = base_id
        obj["_default_anchor"] = anchor
        obj["object_type"] = object_type
        obj["anchor_policy"] = anchor_policy

        if anchor_policy == "stable_part" and not obj.get("stable_anchors"):
            obj["stable_anchors"] = DEFAULT_CONTAINER_ANCHORS if semantic_role in {"destination", "container"} else [anchor]
        if anchor_policy == "dynamic_affordance" and not obj.get("affordance_roles"):
            obj["affordance_roles"] = DEFAULT_DYNAMIC_AFFORDANCES

        obj["stable_anchors"] = _as_list(obj.get("stable_anchors"))
        obj["affordance_roles"] = _as_list(obj.get("affordance_roles"))
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
            object_id, _ = _split_canonical_id(canonical_id)
            lookup[object_id] = obj
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
            obj for obj in objects if obj.get("_semantic_role", obj.get("role")) in ACTIVE_ROLES
        ]

    sequence = []
    seen = set()
    for obj in sequence_objects:
        canonical_id = obj.get("canonical_id")
        if not canonical_id or canonical_id in seen:
            continue
        if obj.get("_semantic_role", obj.get("role")) in NON_TARGET_ROLES:
            warnings.append(
                f"{canonical_id} is context/destination/support and should not be in target_sequence."
            )
        sequence.append(_target_entry(obj))
        seen.add(canonical_id)
    return sequence, warnings


def _first_by_semantic_role(objects: List[Dict[str, Any]], roles: set[str]) -> Dict[str, Any] | None:
    for obj in objects:
        if obj.get("_semantic_role", obj.get("role")) in roles:
            return obj
    return None


def build_context(data: Dict[str, Any], objects: List[Dict[str, Any]]) -> Dict[str, str]:
    context = dict(data.get("context") or {})
    lookup = _object_lookup(objects)
    container = _first_by_semantic_role(objects, {"container", "destination"})
    support_surface = _first_by_semantic_role(objects, {"support_surface"})
    destination_region = _first_by_semantic_role(objects, {"destination_region"})

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
        if obj and nested_key == "destination" and obj.get("_semantic_role") in {"container", "destination"}:
            container_id, _ = _split_canonical_id(obj["canonical_id"])
            context[nested_key] = f"{container_id}:inner_zone"
        elif obj:
            context[nested_key] = obj["canonical_id"]

    return {key: value for key, value in context.items() if value}


def _state_suffix(canonical_id: str) -> str:
    object_id, _ = _split_canonical_id(canonical_id)
    return object_id


def _effect_exists(effects: List[Dict[str, Any]], effect_type: str) -> bool:
    return any(effect.get("effect_type") == effect_type for effect in effects)


def build_expected_effects(
    data: Dict[str, Any],
    objects: List[Dict[str, Any]],
    target_sequence: List[Dict[str, Any]],
    context: Dict[str, str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings = []
    raw_effects = [dict(effect) for effect in data.get("expected_effects") or [] if effect]
    lookup = _object_lookup(objects)
    effects = []
    context = context or {}

    for effect in raw_effects:
        obj_ref = effect.get("object") or effect.get("object_name")
        obj = lookup.get(obj_ref) or lookup.get(normalize_object_id(obj_ref))
        if obj:
            effect["object"] = obj["canonical_id"]
        effect.pop("object_name", None)
        if effect.get("object") and effect.get("effect_type"):
            effects.append({k: v for k, v in effect.items() if v not in ("", None, [])})

    instructions = data.get("task_instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]

    task_type = data.get("task_type")
    active = _first_by_semantic_role(objects, {"manipulated_object", "target", "deformable_object", "labware"})
    container = _first_by_semantic_role(objects, {"container", "destination"})

    needs_shape = task_type in {"garment_folding", "compound_task"} or _contains_folding_language(instructions)
    if needs_shape and active and active.get("object_type") == "deformable" and not (
        _effect_exists(effects, "shape_change") or _effect_exists(effects, "topology_change")
    ):
        effects.append(
            {
                "object": active["canonical_id"],
                "effect_type": "shape_change",
                "from_state": "unfolded",
                "to_state": "folded",
            }
        )

    needs_containment = (
        task_type in {"container_placement", "compound_task"}
        or _contains_container_placement_language(instructions)
    )
    if needs_containment and not _effect_exists(effects, "containment_change"):
        if active and container:
            destination = context.get("destination")
            container_state = _state_suffix(container["canonical_id"])
            effect = {
                "object": active["canonical_id"],
                "effect_type": "containment_change",
                "from_state": f"outside_{container_state}",
                "to_state": f"inside_{container_state}",
            }
            if destination:
                effect["destination"] = destination
            effects.append(effect)
        else:
            warnings.append("Task looks like container placement but lacks an active object or container.")

    return effects, warnings


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


def _find_oracle_fields(value: Any, path: str = "") -> List[str]:
    errors = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ORACLE_FIELDS:
                errors.append(f"Oracle field '{path + key}' is not allowed.")
            errors.extend(_find_oracle_fields(item, f"{path}{key}."))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            errors.extend(_find_oracle_fields(item, f"{path}{idx}."))
    return errors


def validate_preannotation(data: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    warnings = []
    assumptions = []
    errors.extend(_find_oracle_fields(data))

    object_ids = [obj.get("canonical_id") for obj in data.get("objects", []) if obj.get("canonical_id")]
    target_ids = [
        obj.get("canonical_id") for obj in data.get("target_sequence", []) if obj.get("canonical_id")
    ]

    if len(object_ids) != len(set(object_ids)):
        errors.append("objects contain duplicate canonical_id values.")
    if len(target_ids) != len(set(target_ids)):
        errors.append("target_sequence contains duplicate canonical_id values.")

    object_set = set(object_ids)
    target_set = set(target_ids)
    object_by_id = {obj.get("canonical_id"): obj for obj in data.get("objects", [])}

    for canonical_id in target_ids:
        object_id, anchor_id = _split_canonical_id(canonical_id)
        obj = object_by_id.get(canonical_id) or object_by_id.get(f"{object_id}:main_body")
        if canonical_id not in object_set:
            errors.append(f"target_sequence canonical_id '{canonical_id}' is not present in objects.")
        if anchor_id in {"inner_zone", "main_surface"} or (obj and obj.get("role") in NON_TARGET_ROLES):
            errors.append(f"{canonical_id} is context/destination/support and must not be in target_sequence.")
        if anchor_id in DYNAMIC_AFFORDANCE_ANCHORS or (obj and obj.get("anchor_policy") == "dynamic_affordance" and anchor_id != "main_body"):
            errors.append(f"Dynamic affordance anchor '{canonical_id}' must not be used as a strict target.")

    context = data.get("context") or {}
    for key in ("destination", "container", "support_surface"):
        if context.get(key) in target_set:
            errors.append(f"context.{key} should not appear in target_sequence.")

    for effect in data.get("expected_effects", []) or []:
        effect_object = effect.get("object")
        if effect_object and effect_object not in object_set and effect_object not in target_set:
            errors.append(f"expected_effects object '{effect_object}' is not present in objects.")

    if data.get("task_type") == "container_placement" and not (
        context.get("container") or context.get("destination")
    ):
        warnings.append("container_placement requires context.container or context.destination.")

    instructions = data.get("task_instruction") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    if _contains_container_placement_language(instructions):
        if not _effect_exists(data.get("expected_effects", []), "containment_change"):
            warnings.append("Instruction indicates placement into a container but no containment_change effect exists.")

    if data.get("task_type") == "garment_folding":
        if data.get("deformable_anchor_policy") != "dynamic_affordance":
            warnings.append("garment_folding should set deformable_anchor_policy: dynamic_affordance.")
        if not (_effect_exists(data.get("expected_effects", []), "shape_change") or _effect_exists(data.get("expected_effects", []), "topology_change")):
            warnings.append("garment_folding should include shape_change or topology_change.")

    for obj in data.get("objects", []):
        if obj.get("object_type") == "deformable" and obj.get("anchor_policy") != "dynamic_affordance":
            warnings.append(f"{obj.get('canonical_id')} is deformable but anchor_policy is not dynamic_affordance.")
        if obj.get("role") in OVER_SPECIFIC_ROLES:
            warnings.append(
                f"{obj.get('canonical_id')} uses over-specific role '{obj.get('role')}'; prefer object_type/context."
            )

    if data.get("task_type") in {"container_placement", "compound_task"}:
        assumptions.append("Container destinations are represented in context/task_steps, not target_sequence.")

    return {
        "status": "error" if errors else "warning" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "assumptions": assumptions,
    }


def _export_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    export = {
        "canonical_id": obj["canonical_id"],
        "object_name": obj["object_name"],
    }
    if obj.get("color") not in ("", None, "none", "unknown"):
        export["color"] = obj["color"]
    semantic_role = obj.get("_semantic_role", obj.get("role"))
    if semantic_role in {"destination", "tool", "fixture"}:
        export["role"] = semantic_role
    if obj.get("object_type"):
        export["object_type"] = obj["object_type"]
    if obj.get("anchor_policy"):
        export["anchor_policy"] = obj["anchor_policy"]
    if obj.get("stable_anchors"):
        export["stable_anchors"] = obj["stable_anchors"]
    if obj.get("affordance_roles"):
        export["affordance_roles"] = obj["affordance_roles"]
    return export


def _build_task_steps(
    data: Dict[str, Any],
    objects: List[Dict[str, Any]],
    context: Dict[str, str],
    expected_effects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if data.get("task_steps"):
        return data["task_steps"]
    if data.get("task_type") != "compound_task":
        return []
    active = _first_by_semantic_role(objects, {"deformable_object", "manipulated_object", "target"})
    container = _first_by_semantic_role(objects, {"container", "destination"})
    if not active:
        return []
    steps = []
    shape_effects = [effect for effect in expected_effects if effect.get("effect_type") in {"shape_change", "topology_change"}]
    containment_effects = [effect for effect in expected_effects if effect.get("effect_type") == "containment_change"]
    if shape_effects:
        steps.append(
            {
                "step_id": f"fold_{normalize_object_id(active['object_name'])}",
                "task_type": "garment_folding",
                "instruction": f"Fold the {active['object_name']}.",
                "active_objects": [active["canonical_id"]],
                "expected_effects": shape_effects,
            }
        )
    if container and containment_effects:
        steps.append(
            {
                "step_id": f"place_into_{normalize_object_id(container['object_name'])}",
                "task_type": "container_placement",
                "instruction": f"Put the {active['object_name']} into the {container['object_name']}.",
                "active_objects": [active["canonical_id"]],
                "destination": context.get("destination"),
                "expected_effects": containment_effects,
            }
        )
    return steps


def build_preannotation_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_data, oracle_errors = _strip_oracle_fields(deepcopy(data))
    raw_objects = _input_objects(cleaned_data.get("objects") or [])
    task_type = _infer_task_type(cleaned_data, raw_objects)
    cleaned_data["task_type"] = task_type
    cleaned_data["schema_version"] = SCHEMA_VERSION

    objects = build_unique_canonical_ids(raw_objects, task_type)
    target_sequence, target_warnings = build_target_sequence(
        objects,
        cleaned_data.get("target_sequence") or None,
    )
    context = build_context(cleaned_data, objects)
    expected_effects, effect_warnings = build_expected_effects(cleaned_data, objects, target_sequence, context)
    task_steps = _build_task_steps(cleaned_data, objects, context, expected_effects)

    result = {
        "schema_version": SCHEMA_VERSION,
        "task_type": task_type,
        "task_instruction": cleaned_data.get("task_instruction") or [],
        "target_sequence": target_sequence,
        "objects": [_export_object(obj) for obj in objects],
    }
    if context:
        result["context"] = context
    if expected_effects:
        result["expected_effects"] = expected_effects
    if task_steps:
        result["task_steps"] = task_steps

    for passthrough in (
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
    ):
        if passthrough in cleaned_data:
            result[passthrough] = cleaned_data[passthrough]

    has_deformable = any(obj.get("object_type") == "deformable" for obj in objects)
    if task_type in {"garment_folding", "compound_task"} and has_deformable:
        result["deformable_anchor_policy"] = "dynamic_affordance"

    validation = validate_preannotation(result)
    errors = list(dict.fromkeys(oracle_errors + validation["errors"]))
    warnings = list(dict.fromkeys(target_warnings + effect_warnings + validation["warnings"]))
    validation["errors"] = errors
    validation["warnings"] = warnings
    validation["status"] = "error" if errors else "warning" if warnings else "pass"

    if warnings:
        result["warnings"] = warnings
    result["needs_review"] = bool(errors or warnings)
    result["validation"] = validation
    return result

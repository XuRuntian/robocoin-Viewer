import yaml

from src.core.config_generator import ConfigGenerator
from src.core.preannotation import build_preannotation_yaml, validate_preannotation


def test_dynamic_folding_omits_target_sequence_and_redundant_fields():
    result = build_preannotation_yaml(
        {
            "task_type": "garment_folding",
            "task_instruction": ["Fold the blue towel."],
            "objects": [
                {
                    "object_name": "blue towel",
                    "color": "blue",
                    "anchor_policy": "dynamic_affordance",
                },
                {
                    "object_name": "tray",
                    "color": "green",
                    "anchor_policy": "static",
                },
            ],
            "context": {"support_surface": "tray"},
            "expected_effects": [
                {
                    "object": "blue_towel",
                    "effect_type": "shape_change",
                    "to_state": "folded",
                }
            ],
        }
    )

    assert result == {
        "task_type": "garment_folding",
        "task_instruction": "Fold the blue towel.",
        "objects": [
            {"id": "blue_towel", "name": "blue towel", "color": "blue", "anchor_policy": "dynamic_affordance"},
            {"id": "tray", "name": "tray", "color": "green", "anchor_policy": "static"},
        ],
        "context": {"support_surface": "tray:main_body"},
        "expected_effects": [
            {
                "object": "blue_towel:main_body",
                "effect_type": "shape_change",
                "to_state": "folded",
            }
        ],
    }


def test_static_targets_are_normalized_and_ordered():
    result = build_preannotation_yaml(
        {
            "task_type": "labware_rearrangement",
            "task_instruction": "Move the rack, then the beaker.",
            "objects": [
                {"name": "test-tube rack", "color": "white", "anchor_policy": "static"},
                {"name": "beaker", "color": "transparent", "anchor_policy": "static"},
                {"name": "table", "color": "gray", "anchor_policy": "static"},
            ],
            "target_sequence": ["Test-Tube Rack", "beaker"],
            "context": {"support_surface": "table"},
            "expected_effects": [
                {"object": "test-tube rack", "effect_type": "placement_change"},
                {"object": "beaker", "effect_type": "placement_change"},
            ],
        }
    )

    assert result["target_sequence"] == [
        "test_tube_rack:main_body",
        "beaker:main_body",
    ]
    assert result["context"] == {"support_surface": "table:main_body"}


def test_target_sequence_allows_repeated_stable_reference():
    result = build_preannotation_yaml(
        {
            "task_type": "surface_cleaning",
            "task_instruction": "Wipe the counter twice.",
            "objects": [
                {"name": "counter", "color": "gray", "anchor_policy": "static"},
            ],
            "target_sequence": ["counter", "counter"],
            "expected_effects": [
                {"object": "counter", "effect_type": "surface_change"},
            ],
        }
    )

    assert result["target_sequence"] == [
        "counter:main_body",
        "counter:main_body",
    ]
    assert validate_preannotation(result)["status"] == "pass"


def test_context_allows_custom_relationships():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Place the apple near the marker.",
            "objects": [
                {"name": "apple", "color": "red", "anchor_policy": "static"},
                {"name": "marker", "color": "blue", "anchor_policy": "static"},
            ],
            "context": {"reference_marker": "marker:handle"},
            "expected_effects": [
                {"object": "apple", "effect_type": "placement_change"},
            ],
        }
    )

    assert result["context"] == {"reference_marker": "marker:handle"}
    assert validate_preannotation(result)["status"] == "pass"


def test_expected_effect_type_list_is_expanded_to_atomic_records():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move and align the rack.",
            "objects": [
                {"name": "rack", "color": "white", "anchor_policy": "static"},
            ],
            "expected_effects": [
                {
                    "object": "rack",
                    "effect_type": ["placement_change", "alignment_change"],
                },
            ],
        }
    )

    assert result["expected_effects"] == [
        {"object": "rack:main_body", "effect_type": "placement_change"},
        {"object": "rack:main_body", "effect_type": "alignment_change"},
    ]
    assert validate_preannotation(result)["status"] == "pass"


def test_atomic_effect_records_preserve_independent_properties():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move and align the block.",
            "objects": [
                {"name": "block", "color": "red", "anchor_policy": "static"},
                {"name": "counter", "color": "gray", "anchor_policy": "static"},
            ],
            "expected_effects": [
                {
                    "object": "block",
                    "effect_type": "placement_change",
                    "to_state": "on_counter",
                    "destination": "counter:main_body",
                },
                {
                    "object": "block",
                    "effect_type": "alignment_change",
                    "to_state": "aligned",
                },
            ],
        }
    )

    assert result["expected_effects"] == [
        {
            "to_state": "on_counter",
            "destination": "counter:main_body",
            "object": "block:main_body",
            "effect_type": "placement_change",
        },
        {
            "to_state": "aligned",
            "object": "block:main_body",
            "effect_type": "alignment_change",
        },
    ]


def test_static_garment_can_use_stable_sleeve_anchors():
    result = build_preannotation_yaml(
        {
            "task_type": "garment_folding",
            "task_instruction": "Fold the left sleeve, then the right sleeve.",
            "objects": [
                {"name": "jacket", "color": "black", "anchor_policy": "static"},
            ],
            "target_sequence": [
                "Jacket:Wearer-Left-Sleeve",
                "jacket:wearer_right_sleeve",
            ],
            "expected_effects": [
                {"object": "jacket", "effect_type": "shape_change", "to_state": "folded"},
            ],
        }
    )

    assert result["target_sequence"] == [
        "jacket:wearer_left_sleeve",
        "jacket:wearer_right_sleeve",
    ]
    assert validate_preannotation(result)["status"] == "pass"


def test_container_placement_uses_context_without_container_target():
    result = build_preannotation_yaml(
        {
            "task_type": "container_placement",
            "task_instruction": "Place the baozi into the steamer.",
            "objects": [
                {"name": "baozi", "color": "white", "anchor_policy": "static"},
                {"name": "steamer", "color": "silver", "anchor_policy": "static"},
            ],
            "target_sequence": ["baozi"],
            "context": {"container": "steamer", "destination": "steamer:inner-zone"},
            "expected_effects": [
                {"object": "baozi", "effect_type": "containment_change", "to_state": "inside_steamer"},
            ],
        }
    )

    assert result["target_sequence"] == ["baozi:main_body"]
    assert result["context"] == {
        "container": "steamer:main_body",
        "destination": "steamer:inner_zone",
    }
    assert validate_preannotation(result)["status"] == "pass"


def test_same_name_objects_require_explicit_stable_ids():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move both blocks.",
            "objects": [
                {"name": "block", "color": "red", "anchor_policy": "static"},
                {"name": "block", "color": "red", "anchor_policy": "static"},
            ],
            "target_sequence": ["block", "block"],
            "expected_effects": [
                {"object": "block", "effect_type": "placement_change"},
            ],
        }
    )

    assert [obj["id"] for obj in result["objects"]] == ["block", "block"]
    validation = validate_preannotation(result)
    assert validation["status"] == "error"
    assert any("assign an explicit id" in error for error in validation["errors"])


def test_same_name_objects_accept_explicit_stable_ids():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move both blocks.",
            "objects": [
                {"id": "left_block", "name": "block", "color": "red", "anchor_policy": "static"},
                {"id": "right_block", "name": "block", "color": "red", "anchor_policy": "static"},
            ],
            "target_sequence": ["left_block", "right_block"],
            "expected_effects": [
                {"object": "left_block", "effect_type": "placement_change"},
                {"object": "right_block", "effect_type": "alignment_change"},
            ],
        }
    )

    assert [obj["id"] for obj in result["objects"]] == ["left_block", "right_block"]
    assert validate_preannotation(result)["status"] == "pass"


def test_default_id_matching_name_is_exported():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move the test tube rack.",
            "objects": [
                {
                    "id": "test_tube_rack",
                    "name": "test-tube rack",
                    "color": "white",
                    "anchor_policy": "static",
                },
            ],
            "target_sequence": ["test_tube_rack"],
            "expected_effects": [
                {"object": "test_tube_rack", "effect_type": "placement_change"},
            ],
        }
    )

    assert result["objects"] == [
        {"id": "test_tube_rack", "name": "test-tube rack", "color": "white", "anchor_policy": "static"},
    ]


def test_dynamic_affordance_object_cannot_be_strict_target():
    validation = validate_preannotation(
        {
            "task_type": "garment_folding",
            "task_instruction": "Fold the towel.",
            "objects": [
                {"name": "towel", "color": "blue", "anchor_policy": "dynamic_affordance"},
            ],
            "target_sequence": ["towel:corner"],
            "expected_effects": [
                {"object": "towel", "effect_type": "shape_change"},
            ],
        }
    )

    assert validation["status"] == "error"
    assert any("not static" in error for error in validation["errors"])
    assert any("dynamic affordance anchor" in error for error in validation["errors"])


def test_effect_object_must_exist_and_effect_type_is_limited():
    validation = validate_preannotation(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move the apple.",
            "objects": [
                {"name": "apple", "color": "red", "anchor_policy": "static"},
            ],
            "expected_effects": [
                {"object": "banana", "effect_type": "move_change"},
            ],
        }
    )

    assert validation["status"] == "error"
    assert any("unknown object 'banana'" in error for error in validation["errors"])
    assert any("effect_type is not supported" in error for error in validation["errors"])


def test_metadata_passthrough_is_preserved():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move the apple.",
            "objects": [
                {"name": "apple", "color": "red", "anchor_policy": "static"},
            ],
            "expected_effects": [
                {"object": "apple", "effect_type": "placement_change"},
            ],
            "dataset_batch_number": 3,
            "scene_level1": "Lab",
            "atomic_actions": ["grasp", "place"],
            "device_model": ["Galbot_G1"],
        }
    )

    assert result["dataset_batch_number"] == 3
    assert result["scene_level1"] == "Lab"
    assert result["atomic_actions"] == ["grasp", "place"]
    assert result["device_model"] == ["Galbot_G1"]


def test_oracle_fields_are_removed_and_generated_yaml_can_be_loaded():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": "Move the apple.",
            "frame": 10,
            "objects": [
                {
                    "name": "apple",
                    "color": "red",
                    "anchor_policy": "static",
                    "timestamp": 123.4,
                }
            ],
            "expected_effects": [
                {"object": "apple", "effect_type": "placement_change"},
            ],
        }
    )

    yaml_text = ConfigGenerator.generate_yaml_string(result)
    loaded = yaml.safe_load(yaml_text)
    assert "frame" not in loaded
    assert "timestamp" not in loaded["objects"][0]
    assert loaded["expected_effects"][0]["object"] == "apple:main_body"

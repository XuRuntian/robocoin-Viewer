import yaml

from src.core.config_generator import ConfigGenerator
from src.core.preannotation import build_preannotation_yaml, validate_preannotation


def test_container_destination_is_not_auto_target():
    result = build_preannotation_yaml(
        {
            "task_instruction": ["Place the baozi into the steamer."],
            "objects": ["baozi", "steamer"],
        }
    )

    assert result["schema_version"] == "task_prior_v1_5"
    assert result["task_type"] == "container_placement"
    assert result["target_sequence"] == [
        {"canonical_id": "baozi:main_body", "object_name": "baozi"}
    ]
    assert result["objects"][1]["role"] == "destination"
    assert result["context"]["destination"] == "steamer:inner_zone"
    assert result["context"]["container"] == "steamer:main_body"


def test_baozi_into_steamer_generates_containment_change():
    result = build_preannotation_yaml(
        {
            "task_type": "container_placement",
            "task_instruction": ["Place the baozi into the steamer."],
            "objects": [
                {"object_name": "baozi"},
                {"object_name": "steamer"},
            ],
        }
    )

    assert result["objects"][0]["object_type"] == "soft_rigid"
    assert result["objects"][0]["anchor_policy"] == "none"
    assert result["objects"][1]["object_type"] == "rigid"
    assert result["objects"][1]["anchor_policy"] == "stable_part"
    assert result["expected_effects"] == [
        {
            "object": "baozi:main_body",
            "effect_type": "containment_change",
            "from_state": "outside_steamer",
            "to_state": "inside_steamer",
            "destination": "steamer:inner_zone",
        }
    ]
    assert result["validation"]["status"] == "pass"


def test_garment_folding_generates_dynamic_affordance_and_shape_change():
    result = build_preannotation_yaml(
        {
            "task_type": "garment_folding",
            "task_instruction": ["Fold the blue towel."],
            "objects": [
                {
                    "object_name": "blue towel",
                    "color": "blue",
                }
            ],
        }
    )

    assert result["deformable_anchor_policy"] == "dynamic_affordance"
    assert result["target_sequence"] == [
        {"canonical_id": "blue_towel:main_body", "object_name": "blue towel"}
    ]
    assert result["objects"][0]["object_type"] == "deformable"
    assert result["objects"][0]["anchor_policy"] == "dynamic_affordance"
    assert result["objects"][0]["affordance_roles"] == [
        "corner",
        "edge",
        "crease",
        "fold_region",
        "contact_patch",
    ]
    assert result["expected_effects"] == [
        {
            "object": "blue_towel:main_body",
            "effect_type": "shape_change",
            "from_state": "unfolded",
            "to_state": "folded",
        }
    ]


def test_compound_task_supports_task_steps_without_repeating_target():
    result = build_preannotation_yaml(
        {
            "task_type": "compound_task",
            "task_instruction": ["Fold the blue towel and put it into the basket."],
            "objects": [
                {"object_name": "blue towel", "color": "blue"},
                {"object_name": "basket"},
            ],
        }
    )

    assert result["task_type"] == "compound_task"
    assert result["deformable_anchor_policy"] == "dynamic_affordance"
    assert result["target_sequence"] == [
        {"canonical_id": "blue_towel:main_body", "object_name": "blue towel"}
    ]
    assert [step["step_id"] for step in result["task_steps"]] == [
        "fold_blue_towel",
        "place_into_basket",
    ]
    assert result["task_steps"][1]["destination"] == "basket:inner_zone"


def test_duplicate_objects_are_numbered():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": ["Place two baozi on the table."],
            "objects": ["baozi", "baozi", "table"],
        }
    )

    object_ids = [obj["canonical_id"] for obj in result["objects"]]
    target_ids = [obj["canonical_id"] for obj in result["target_sequence"]]
    assert "baozi_1:main_body" in object_ids
    assert "baozi_2:main_body" in object_ids
    assert target_ids == ["baozi_1:main_body", "baozi_2:main_body"]


def test_oracle_fields_are_validation_errors_and_removed_from_result():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": ["Pick the apple."],
            "frame": 10,
            "objects": [
                {
                    "object_name": "apple",
                    "timestamp": 123.4,
                }
            ],
        }
    )

    assert "frame" not in result
    assert "timestamp" not in result["objects"][0]
    assert result["validation"]["status"] == "error"
    assert any("oracle field" in error.lower() for error in result["validation"]["errors"])


def test_dynamic_affordance_anchor_cannot_be_target():
    validation = validate_preannotation(
        {
            "schema_version": "task_prior_v1_5",
            "task_type": "garment_folding",
            "objects": [
                {
                    "canonical_id": "blue_towel:main_body",
                    "object_name": "blue towel",
                    "object_type": "deformable",
                    "anchor_policy": "dynamic_affordance",
                }
            ],
            "target_sequence": [
                {"canonical_id": "blue_towel:corner", "object_name": "blue towel"}
            ],
            "expected_effects": [],
        }
    )

    assert validation["status"] == "error"
    assert any("Dynamic affordance anchor" in error for error in validation["errors"])


def test_expected_effect_object_must_exist():
    validation = validate_preannotation(
        {
            "schema_version": "task_prior_v1_5",
            "task_type": "pick_and_place",
            "objects": [
                {"canonical_id": "apple:main_body", "object_name": "apple"}
            ],
            "target_sequence": [
                {"canonical_id": "apple:main_body", "object_name": "apple"}
            ],
            "expected_effects": [
                {"object": "banana:main_body", "effect_type": "placement_change"}
            ],
        }
    )

    assert validation["status"] == "error"
    assert any("banana:main_body" in error for error in validation["errors"])


def test_generated_yaml_can_be_safe_loaded():
    result = build_preannotation_yaml(
        {
            "task_instruction": ["Place the baozi into the steamer."],
            "objects": ["baozi", "steamer"],
        }
    )

    yaml_text = ConfigGenerator.generate_yaml_string(result)
    loaded = yaml.safe_load(yaml_text)
    assert loaded["schema_version"] == "task_prior_v1_5"
    assert loaded["context"]["destination"] == "steamer:inner_zone"

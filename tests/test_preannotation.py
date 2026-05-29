from src.core.preannotation import build_preannotation_yaml, validate_preannotation


def test_container_is_not_auto_target_and_destination_uses_inner_zone():
    data = {
        "task_type": "container_placement",
        "task_instruction": ["Place the steamed bun into the steamer."],
        "objects": [
            {"object_name": "steamed bun", "role": "manipulated_object"},
            {"object_name": "steamer", "role": "container"},
            {"object_name": "table", "role": "support_surface"},
        ],
    }

    result = build_preannotation_yaml(data)

    assert result["target_sequence"] == [
        {
            "canonical_id": "steamed_bun:main_body",
            "object_name": "steamed bun",
            "role": "manipulated_object",
        }
    ]
    assert result["context"]["destination"] == "steamer:inner_zone"
    assert result["context"]["container"] == "steamer:main_body"
    assert result["context"]["support_surface"] == "table:main_surface"


def test_container_placement_generates_containment_change():
    data = {
        "task_type": "container_placement",
        "task_instruction": ["Place the steamed bun into the steamer."],
        "objects": [
            {"object_name": "steamed bun", "role": "manipulated_object"},
            {"object_name": "steamer", "role": "container"},
        ],
    }

    result = build_preannotation_yaml(data)

    assert result["expected_effects"] == [
        {
            "object": "steamed_bun:main_body",
            "effect_type": "containment_change",
            "from_state": "outside_steamer",
            "to_state": "inside_steamer",
        }
    ]
    assert result["needs_review"] is False


def test_duplicate_objects_are_numbered():
    data = {
        "task_type": "pick_and_place",
        "task_instruction": ["Place two steamed buns on the table."],
        "objects": [
            {"object_name": "steamed bun", "role": "manipulated_object"},
            {"object_name": "steamed bun", "role": "manipulated_object"},
            {"object_name": "table", "role": "support_surface"},
        ],
    }

    result = build_preannotation_yaml(data)

    object_ids = [obj["canonical_id"] for obj in result["objects"]]
    target_ids = [obj["canonical_id"] for obj in result["target_sequence"]]
    assert "steamed_bun_1:main_body" in object_ids
    assert "steamed_bun_2:main_body" in object_ids
    assert target_ids == ["steamed_bun_1:main_body", "steamed_bun_2:main_body"]


def test_target_sequence_id_must_exist_in_objects():
    validation = validate_preannotation(
        {
            "task_type": "pick_and_place",
            "objects": [
                {
                    "canonical_id": "apple:main_body",
                    "object_name": "apple",
                    "role": "manipulated_object",
                }
            ],
            "target_sequence": [
                {
                    "canonical_id": "banana:main_body",
                    "object_name": "banana",
                    "role": "manipulated_object",
                }
            ],
            "expected_effects": [],
        }
    )

    assert validation["status"] == "warning"
    assert any("banana:main_body" in warning for warning in validation["warnings"])


def test_oracle_fields_are_removed_and_warned():
    result = build_preannotation_yaml(
        {
            "task_type": "pick_and_place",
            "task_instruction": ["Pick the apple."],
            "frame": 10,
            "objects": [
                {
                    "object_name": "apple",
                    "role": "manipulated_object",
                    "timestamp": 123.4,
                }
            ],
        }
    )

    assert "frame" not in result
    assert "timestamp" not in result["objects"][0]
    assert result["needs_review"] is True
    assert any("oracle field" in warning for warning in result["warnings"])


def test_deformable_policy_only_for_garment_folding():
    garment = build_preannotation_yaml(
        {
            "task_type": "garment_folding",
            "task_instruction": ["Fold the towel."],
            "objects": [
                {"object_name": "towel", "role": "deformable_object"},
            ],
        }
    )
    container = build_preannotation_yaml(
        {
            "task_type": "container_placement",
            "task_instruction": ["Place the steamed bun into the steamer."],
            "objects": [
                {"object_name": "steamed bun", "role": "manipulated_object"},
                {"object_name": "steamer", "role": "container"},
            ],
        }
    )

    assert garment["deformable_anchor_policy"] == "dynamic_affordance"
    assert "deformable_anchor_policy" not in container

from piplan.legacy_expert.expert_dataset_converter import ExpertDatasetConverter


def test_joint_legacy_row_converts_to_pi05_example():
    row = {
        "schema_version": "joint_bc_v1",
        "observation": {
            "joint_state_vector": [20, 20, 1, 0, 1, 1],
            "map": {"width": 20, "height": 20},
            "agents": [{"id": 0, "x": 1.0, "y": 1.0, "destination": {"x": 2.0, "y": 2.0}}],
        },
        "language": {"task_text": "plan test"},
        "action": {"joint_expert_action": [0.1, 0.2]},
    }

    examples = ExpertDatasetConverter(include_images=True).convert([row])

    assert len(examples) == 1
    assert examples[0]["state"] == [20, 20, 1, 0, 1, 1]
    assert examples[0]["action"] == [0.1, 0.2]
    assert examples[0]["image"].shape == (3, 224, 224)

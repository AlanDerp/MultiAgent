from piplan.observation.observation_bundle import Pi05ObservationBuilder
from piplan.runtime.world import make_demo_world


def test_observation_bundle_has_pi05_fields():
    world = make_demo_world(agent_count=10).state
    bundle = Pi05ObservationBuilder().build(world)

    assert bundle.global_state["schema_version"] == "piplan_render_state_v1"
    assert bundle.agent_order == list(range(10))
    assert len(bundle.state_vector) == 163
    assert bundle.agent_tokens.shape == (10, 16)
    assert bundle.state_vector[:3] == [20.0, 20.0, 10.0]
    assert bundle.state_vector[3 + 11] == 1.0
    assert bundle.state_vector[3 + 12] == 1.0
    assert bundle.state_vector[3 + 14] == 1.0
    assert bundle.state_vector[3 + 15] == -1.0
    assert bundle.image.shape == (3, 224, 224)
    assert "Plan coordinated collision-free" in bundle.task

from piplan.policy.action_decoder import JointActionDecoder


def test_action_decoder_maps_pairs_to_agents():
    chunk = JointActionDecoder(
        action_dt=0.1,
        default_speed=2.0,
        output_mode="normalized_tanh",
    ).from_raw([[0.0, 1.0, -1.0, 0.0]], [3, 7])

    assert chunk.agent_order == [3, 7]
    assert chunk.dt == 0.1
    assert chunk.actions[0][3].vx == 0.0
    assert chunk.actions[0][3].vy > 1.5
    assert chunk.actions[0][7].vx < -1.5
    assert chunk.actions[0][7].vy == 0.0


def test_action_decoder_clamps_raw_velocity_mode():
    chunk = JointActionDecoder(action_dt=0.1, default_speed=1.5).from_raw([[2.0, -2.0]], [3])

    assert chunk.actions[0][3].vx == 1.5
    assert chunk.actions[0][3].vy == -1.5

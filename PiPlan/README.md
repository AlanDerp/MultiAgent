# PiPlan

PiPlan is the pi0.5-first refactor of `6000_Dorabot`.

The online planning path is deliberately simple:

```text
WorldState
  -> GlobalStateEncoder
  -> observation.state + observation.images.topdown + task text
  -> pi0.5 action chunk
  -> HorizonBuffer
  -> SafetySupervisor
  -> Actuator
  -> WorldState
```

Traditional `global_planners` and `local_planners` are not part of the PiPlan
runtime. They are available only through `piplan.legacy_expert`, which runs the
old `6000_Dorabot` project as an isolated data-generation source.

## Smoke Run

```bash
cd PiPlan
python -m piplan.cli run-demo --steps 20 --agents 4
```

## Real pi0.5 Run

```bash
cd PiPlan
python scripts/run_pi05_sim.py \
  --mode pi05 \
  --model_id lerobot/pi05_base \
  --lerobot_path ../lerobot/src
```

The real pi0.5 path expects LeRobot pi dependencies to be installed in the
active environment:

```bash
cd ../lerobot
pip install -e ".[pi]"
```

## Legacy Expert Data

Collect traditional expert demonstrations from `6000_Dorabot`:

```bash
cd PiPlan
python scripts/collect_expert_data.py \
  --dorabot-root ../6000_Dorabot \
  --output data/expert/raw/rrt_vf.jsonl \
  --gp RRTStar \
  --lp VirtualForcePlanner \
  --record-mode joint_bc
```

Convert them to a LeRobot dataset:

```bash
python scripts/convert_expert_to_lerobot.py \
  --input data/expert/raw/rrt_vf.jsonl \
  --output data/expert/lerobot/dorabot_joint \
  --repo-id local/piplan-dorabot-joint \
  --overwrite
```

## Migration Boundaries

- `piplan.runtime`: new pi0.5-first runtime loop.
- `piplan.observation`: pi0.5 observation schema and adapters.
- `piplan.policy`: mock and real pi0.5 policy clients.
- `piplan.control`: safety constraints only, not traditional local planning.
- `piplan.legacy_expert`: the only place allowed to call old global/local expert planners.

# Daksha VLA UI

Training & inference web UI for Daksha, built on LeRobot. Full UI docs: [ui/ui/README.md](ui/ui/README.md).

## Environment

This folder holds code only. It runs in the shared workspace environment `~/uni/daksha/robot_venv`, together with all robot packages. See the workspace [README](../../../README.md).

```
vla_codes/
  app.py, ui/         Daksha UI code   ─┐ installed editable into robot_venv:
  lerobot/            LeRobot repo     ─┘ changes apply without reinstalling
  pyproject.toml      the UI's own libraries (ranges)
```

```bash
cd ~/uni/daksha
bash setup_env.sh                           # robot_venv = uv.lock (robot + LeRobot + this UI)
source env.sh
colcon build --packages-select daksha_vla_ui
ros2 launch daksha_vla_ui app.launch.py     # runs app.py inside robot_venv automatically
```

## Changing versions

| What | Where |
|---|---|
| A UI library (Flask, fastapi, …) | `dependencies` in this folder's `pyproject.toml` |
| A LeRobot library | `lerobot/pyproject.toml` (upstream ranges, normally leave it) |
| Which policies are installed | the `lerobot[...]` extras in `~/uni/daksha/pyproject.toml` |
| torch / torchvision / torchcodec | `~/uni/daksha/pyproject.toml` (cu130 index; also set in `lerobot/pyproject.toml`) |

Then run `bash ~/uni/daksha/setup_env.sh --upgrade` and commit `pyproject.toml` and `uv.lock`.

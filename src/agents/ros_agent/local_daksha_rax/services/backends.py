"""Robot backends behind the agent MCP (the RAX backend contract). The frontend never
imports a backend directly; it constructs the registered class once per session.

    tools() -> list[dict]            MCP tool definitions (name, description, inputSchema)
    validate(name, arguments)        raise before any motion
    call(name, arguments) -> (value, images)
    status() -> dict                 live robot summary merged into the `status` tool
    close()                          release resources; safe to call twice

Simulation backends also implement the episode hooks the frontend drives:
reset_episode(seed), end_episode(), task_state().

To add the real Daksha later, register e.g. 'daksha_ros': 'services.daksha_ros.backend:DakshaRosBackend'
with the same tool names, so controllers written in simulation run unchanged.
"""
from importlib import import_module

BACKENDS = {
    'daksha_sim': 'services.daksha_sim.backend:DakshaSimBackend',
    # Same robot tools + `vlm_query` served by local Ollama (qwen3-vl:8b); local_daksha_rax only.
    'daksha_sim_ollama': 'services.daksha_sim.ollama_backend:DakshaSimOllamaBackend',
    # The REAL robot over ROS 2: perception from ros_agent, motion through aruco_hand_eye's
    # aruco_reach actions (the only /joint_cmd sender; dry run until its execute is on).
    # Needs a sourced ROS 2 workspace and `ros2 launch aruco_hand_eye aruco_reach.launch.py`.
    'daksha_ros': 'services.daksha_ros.backend:DakshaRosBackend',
}


def backend_class(name):
    try:
        module, _, attribute = BACKENDS[name].partition(':')
    except KeyError:
        raise ValueError(f"Unknown robot backend {name!r}; choose one of {', '.join(sorted(BACKENDS))}") from None
    return getattr(import_module(module), attribute)


def make_backend(config):
    return backend_class(config.robot)(config)

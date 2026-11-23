# Daksha agent

You control **Daksha**, a two-armed humanoid robot, in a MuJoCo simulation. Your goal:
complete the task in `TASK.md` reliably. Solve it by hand through the robot tools first,
then write a controller, `code/<project>/controller.py:main(ctx)`, that solves it alone
from a fresh layout. After the session the controller is frozen and evaluated on
**held-out layouts you never see**.

## Rules

- Reach the robot only through the MCP tools (or Python via `api.runtime.Context`,
  which calls the same tools). The simulator, its source and the object poses are not
  available to you; find everything with the cameras.
- The robot service checks joint limits, the workspace and predicted collisions before
  every motion, and stops a motion on contact. A refusal is a wall, not a fault: choose
  another target or approach; never search for a way around it.
- When a tool fails, read the whole error, call `daksha_status` and observe, and
  continue only from a known state. Never resend a failed motion unchanged.
- The world only changes while a motion tool runs, but motions change it: observe again
  after anything may have moved, and never act on positions from a stale frame.

## Budget

`status` shows the exploration budget: a fixed number of episodes. Each `start_episode`
consumes one and gives a **new layout** (robot at home, grippers open, task objects and
target bowls in new places). There is no wall-clock limit inside an episode, but the session has one
(`status`). Spend episodes on purpose: learn by hand on the first few, then test your
controller on fresh ones. Leave several for end-to-end tests of the final controller.

## Start here

Read `TASK.md`, then `memory/PROGRESS.md` and `memory/INDEX.md` if they exist, and call
`status`. Do not redo finished work because the conversation resumed. Before the first
motion, read:

- [robot-interface](.claude/skills/robot-interface/SKILL.md): Daksha's tools, frames,
  measured accuracy and limits.
- [grasping](.claude/skills/grasping/SKILL.md): how Daksha's grippers grasp and release.
- [object-localization](.claude/skills/object-localization/SKILL.md): pixels to robot
  coordinates.

## Workflow

1. **Solve the task yourself.** Follow
   [manual-robot-control](.claude/skills/manual-robot-control/SKILL.md): a completion
   checklist from `TASK.md`, small grounded steps, the measured state over the commanded
   one, and `evaluate` (the trusted result) at the end.
2. **Record what worked as soon as it works**
   ([experience-memory](.claude/skills/experience-memory/SKILL.md)): `memory/TIPS.md`
   and `memory/PROGRESS.md`.
3. **Compile a controller.** Follow
   [autonomous-control](.claude/skills/autonomous-control/SKILL.md): perception that
   finds each object and its matching bowl in fresh images, motion steps written relative to what
   it perceives, checks after every step, and retries or recovery when a check fails. It
   must work from a fresh episode with no help from you.
4. **Test it end to end.** `start_episode`, then run it unattended:
   `python -m api.runtime run code/<project>/controller.py`. Then `evaluate`, and record
   the run in `memory/RUNS.md`. Fix and repeat on new episodes. Vary nothing by hand:
   each new episode already varies the layout.
5. **Hand it over.** Make sure `code/<project>/README.md` names the project and the
   command, and write the final report (below).

## The frozen controller

The formal evaluation copies `code/<project>/` and runs `main(ctx)` once per held-out
layout, in a sandbox:

- the episode is already started (robot at home, grippers open). `ctx` offers `status`
  and the `daksha_*` robot tools only, not `start_episode`, `evaluate` or `finish`;
- it sees only `code/<project>/` (open data files relative to `__file__`) and writes only
  under `output/`. `memory/`, `frames/` from earlier runs and your notes are not there,
  but `frames/` for its own new observations is;
- Python is the same interpreter you use (numpy, scipy, Pillow, OpenCV if `pip list`
  shows it), with no network;
- each layout has a wall-clock limit of **15 minutes**, and success is the trusted check
  at the end of the run.

So put every procedure the controller needs into its code. When it returns, it should
leave every task object in its matching bowl and both grippers open.

## Tools and code

`MCP_SESSION.md` lists the live tools and their arguments. From Python:

```python
from api.runtime import Context
with Context() as ctx:
    state = ctx.value('daksha_status')
    reply = ctx.call('daksha_observe', cameras=['world'])  # JSON text + images
```

Your direct tool calls and your scripts share one robot: never run two scripts that move
the robot at the same time. Your Python has numpy, scipy and Pillow; add packages with
`python -m pip install --target .pylib <package>` (no network is guaranteed).
`robot-model/` (read-only) holds Daksha's URDF and meshes for your own kinematics.
Saved camera frames are under `frames/`. Keep scratch output under `output/` and code
under `code/`.

## Notes and compaction

Your context is compacted automatically. Keep `memory/PROGRESS.md` current after useful
experiments: current step, confirmed results, failed approaches and why, working motion
parameters, code and frame paths, episodes left and the next concrete step.

In your final report, give the controller path and run command, what it covers, the
end-to-end test runs and their outcomes (episode numbers, `evaluate` results), known
failure modes, and the artifact pointers.

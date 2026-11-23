# Bionic Hand Control — User Manual

This page connects to the physical bionic hand hardware and lets you control both the left and right hands from a web browser.

Both hands are wired to **one single USB/serial cable** — there is only one connection to make, and it controls all 16 motors across both hands at once (left hand = motor IDs 1–8, right hand = motor IDs 9–16, four fingers per hand with two motors each).

## Opening the App

Open the control page in your browser. The header shows:

- **Title and connection status** — reads "Not connected" or "Connected · <port>".
- **Movement lock status** — "Movement locked" or "Movement enabled", so you can tell at a glance whether commands will actually move the hands.
- **STOP** — the red button in the top-right corner. Press it at any time to immediately halt whatever motion is in progress. Motor power stays on; it only stops the current movement.

---

## 1. Connecting to the Hand (Step by Step)

Do this before anything else — nothing on the page will move the hand until you've connected.

1. **Power on the hand and plug in the cable.** Make sure the hand's power supply is switched on, then connect its USB/serial cable to the computer running this web page. Both hands move over this one cable, so there's nothing further to plug in for the second hand.
2. **Open the Serial port dropdown** in the "Connection & Safety" panel. The hand's cable normally shows up as `/dev/ttyACM0`, which is selected by default.
   - If you don't see the right port listed, click **Refresh** to rescan.
   - If it still doesn't appear, check that the cable is firmly seated and the hand is powered on.
   - If the port is listed but connecting fails, make sure no other program or browser tab is already connected to it — only one connection to the hand is allowed at a time.
3. **Leave Baud at its default** (1,000,000) unless you specifically know the hand's controller is set to a different speed.
4. **Click Connect.** The header status changes to "Connected · <port>" once successful. Any problem connecting is reported both as an on-screen error and as an entry in the Activity Log at the bottom of the page.
5. **Click Ping motors.** This checks all 16 motors and reports which ones (if any) didn't respond — a quick way to confirm the hand is properly wired and powered before you try to move it.
6. **Click Read positions.** This reads the fingers' actual current angles so the app knows the hand's real starting pose. Do this every time right after connecting (and again any time the fingers may have been moved by hand) — the Smooth and Staged motion styles need this starting point to move cleanly.
7. **Switch on Enable movement.** This is a separate safety lock from being connected — until it's on, gestures, finger sliders, wave, and the motor test are all blocked. Treat it as the final "arm the system" step right before you intend to move the hand.

You're now ready to use the gestures, finger sliders, wave, and single motor test described below.

**To disconnect:** click **Disconnect**. This stops any motion, frees the connection for another program, and automatically re-locks movement, so you'll need to switch Enable movement back on the next time you connect.

---

## 2. Connection & Safety Panel Reference

- **Serial port** — the port the hand is connected on (see the connection steps above).
- **Refresh** — rescans for available serial ports and updates the list.
- **Baud** — communication speed (default 1,000,000). Only change this if your hardware is configured differently.
- **Connect / Disconnect** — connects to or disconnects from the selected port. Disconnecting automatically locks movement.
- **Enable movement** (toggle, top right of the panel) — a safety lock. Even once connected, no gesture, finger move, wave, or motor test will run until this is switched on.
- **Ping motors** — sends a check to all 16 motors and reports which ones responded.
- **Read positions** — asks the hand to report where each motor currently is.

---

## 3. Motion Settings

These settings apply to every movement you trigger from the gestures, wave, and finger panels below.

- **Speed** (1–1000) — how fast the motors move.
- **Duration** (0–20 seconds) — how long a motion should take to complete.
- **Motion style** — how the fingers travel to their target position:
  - **Auto** — left hand moves finger-by-finger (staged), right hand moves smoothly. This is the default.
  - **Direct** — the motor jumps straight to the target at the chosen speed, with no interpolation.
  - **Smooth** — moves in small incremental steps for a fluid motion.
  - **Staged** — moves one finger at a time in sequence.

  Note: Smooth and Staged need to know the hand's current position first. If the position isn't known yet, the app will automatically try to read it; if that also fails, it falls back to Direct motion for that one move.

---

## 4. Left / Right Hand Panels

Each hand panel (Left hand, Right hand) shows the four fingers: **Thumb, Index, Middle, Ring**, each with its motor ID pair.

- **Position known / Position unknown** label — shows whether the app currently has a valid reading of that hand's position. Use "Read positions" in the Connection panel to refresh this.
- **Checkbox per finger** — select which fingers are included in the next "Move selected fingers" command. Unchecked fingers are left alone.
- **Select all** — ticks every finger's checkbox at once.
- **Slider per finger** — set a target position from 0% (fully open) to 100% (fully closed) for that finger. The grey bar under the slider shows the hand's current commanded position for comparison.
- **Open / Close buttons** (per finger) — instantly set that single finger's slider to 0% or 100%.
- **Move selected fingers** — sends only the checked fingers to their slider positions, using the current Speed/Duration/Motion style settings.
- **All open** — opens every finger on that hand.
- **All closed** — closes every finger on that hand.
- **Motor ID readout row** — shows the live target angle (in degrees) for each of the 8 motor IDs on that hand, for reference.

---

## 5. Gestures

Pick a preset hand shape and apply it instantly.

- **Left / Right / Both** toggle (with the "Both hands at the same time" switch) — choose which hand(s) the gesture applies to. When "Both hands at the same time" is on, both hands perform the gesture simultaneously rather than one after another.
- **Gesture buttons** — available presets:
  - Open Hand
  - Close / Fist
  - Thumbs Up
  - Point
  - Peace / V
  - Victory / V (Thumb Open)
  - Three Fingers
  - Middle Finger Closed
  - Index Closed
  - Index + Ring
  - Middle + Ring

  Clicking a button runs that gesture using the current Motion Settings (speed, duration, motion style).

---

## 6. Wave

Plays a repeated open/close "wave" motion.

- **Cycles** — how many times the hand opens and closes (1–10).
- **Left only** — waves the left hand only.
- **Right only** — waves the right hand only.
- **Left → Right** — left hand waves first, then the right hand.
- **Right → Left** — right hand waves first, then the left hand.
- **Both together** — both hands wave at the same time.

---

## 7. Single Motor Test

For testing or fine-tuning one motor directly, bypassing gestures and finger groupings.

- **Motor ID** — the individual motor to move. IDs 1–8 belong to the left hand, IDs 9–16 to the right hand.
- **Angle (0–300)** — the target angle in degrees to send to that motor.
- **Send** — moves the specified motor to the specified angle, using the current Speed setting.

---

## 8. Activity Log

A running log at the bottom of the page records every action as it happens:

- **Started: …** — a command has been sent and is in progress.
- **Done: …** (shown in green) — the command completed successfully.
- **Stopped: …** (shown in yellow/orange) — the movement was interrupted (e.g., by pressing STOP); the hand's exact position is no longer guaranteed to be known, so consider using "Read positions" afterward.
- **Error: …** (shown in red) — something went wrong (e.g., not connected, or a motor didn't respond).

Only one movement can run at a time. If you trigger a new action while another is still in progress, you'll see a "Busy" message — wait for the current action to finish or press STOP first.

---

## Typical Workflow

1. Power on the hand, plug in the cable, select its serial port (`/dev/ttyACM0` by default), and click **Connect**.
2. Click **Ping motors** to confirm all motors respond.
3. Click **Read positions** so the app knows the hand's starting pose.
4. Switch on **Enable movement**.
5. Adjust **Speed**, **Duration**, and **Motion style** as desired.
6. Use **Gestures**, the **finger sliders**, or **Wave** to move the hand.
7. Press **STOP** at any time to halt motion immediately.
8. When finished, switch off **Enable movement** and click **Disconnect** before unplugging the cable.

---

## Troubleshooting

- **Can't find the port / Connect fails** — check the cable is plugged in firmly and the hand's power supply is on. Click **Refresh** to rescan. Make sure no other program or browser tab already has a connection open — only one connection to the hand is allowed at a time.
- **"Movement is locked" message** — you're connected but haven't switched on **Enable movement** yet.
- **A motor doesn't respond to Ping** — check that motor's wiring and power; a missing ID usually means a loose connector on that finger.
- **"Position unknown" on a hand** — click **Read positions** for that hand, or use **All open** / **All closed** once first, before relying on Smooth or Staged motion.
- **"Busy" message when starting a new action** — another movement is still running. Wait for it to finish or press **STOP**.
- **Need to hand off control to another program** — click **Disconnect** first so the port is free.

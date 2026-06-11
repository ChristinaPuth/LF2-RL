# LF2-RL: Training a Reinforcement Learning Agent to Play Little Fighter 2

> **Based on [F.LF](https://github.com/Project-F/F.LF)** — an open-source HTML5 reimplementation of Little Fighter 2 by Project-F. The game engine, character data, and JS controller are inherited from that project. We added a WebSocket bridge and RL training code on top.

Training an RL agent (Davis) to defeat the built-in AI opponent (Dennis) in Little Fighter 2, a classic 2.5D fighting game.

The agent went through two training phases, achieving a **50–64% rolling win rate** against the hard-difficulty AI in the final phase.

---

## Results

| Phase | Method | Win Rate (Hard) |
|-------|--------|-----------------|
| Phase 2 | Custom Env + PPO | 9.6% |
| Phase 3 | Double DQN + Curriculum | **50–64%** |

---

## Repository Structure

```
lf2-rl/
├── lf2_rl/
│   ├── env.py           # Gymnasium environment — WebSocket bridge, reward, obs
│   ├── bridge.py        # Low-level WebSocket server (JS ↔ Python communication)
│   ├── train.py         # Phase 2: PPO training loop (Stable-Baselines3)
│   ├── train_DQN.py     # Phase 3: Double DQN training loop
│   ├── test_bridge.py   # Bridge connection test utility
│   └── logs/            # Training logs and win-rate plots
├── LF/                  # F.LF game engine (inherited, with RL modifications)
│   ├── match.js         # Match logic (modified to push state via WebSocket)
│   └── controller.js    # Key injection controller
└── index.html           # Game entry point
```

All training code lives in `lf2_rl/`. Use `train.py` for Phase 2 (PPO) and `train_DQN.py` for Phase 3 (Double DQN).

---

## How It Works

The game runs in a browser (F.LF). A Python Gym environment communicates with it over WebSocket:

```
Browser (F.LF JS game)
        ↕  WebSocket (localhost:8765)
Python LF2Env (env.py)
        ↕  step() / reset()
RL Agent (Double DQN)
        ↓  action (0–10) injected via JS controller
```

Each game frame, the JS side pushes a JSON state object (HP, MP, position, facing direction for both characters) to Python. The agent picks one of 11 discrete macro-actions; the controller injects the corresponding key inputs.

---

## Quickstart

**1. Install dependencies**
```bash
pip install torch numpy gymnasium websockets stable-baselines3
```

**2. Serve the game**
```bash
cd /path/to/repo
python -m http.server 8000
```

**3. Open the game in browser**
```
http://localhost:8000/index.html
```

**4. Run training**
```bash
cd lf2_rl

# Phase 2: PPO
python train.py

# Phase 3: Double DQN (recommended)
python train_DQN.py
```

The script waits for the browser to connect, then starts training automatically.

---

## Action Space

11 discrete macro-actions:

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | Idle | 6 | Defend |
| 1 | Left | 7 | Left + Attack |
| 2 | Right | 8 | Right + Attack |
| 3 | Jump | 9 | Jump + Attack |
| 4 | Crouch | 10 | Down + Attack (Shoryuken) |
| 5 | Attack | | |

Special moves are packed as single atomic actions — no key-sequence learning required.

---

## Key Design Decisions

- **Off-policy (Double DQN + Replay Buffer)**: each experience is reused many times, critical when browser steps are slow
- **Macro-actions**: special moves packed as single actions to avoid needing recurrent memory
- **Anti-passivity reward**: per-step penalty to prevent stalling strategies
- **Timeout penalty > loss penalty**: prevents the agent from dragging out losing matches
- **3-stage curriculum**: Dumbass AI → Crusher AI → hardest AI, advancing on 70% rolling win rate

---

## Acknowledgements

This project builds on [F.LF](https://github.com/Project-F/F.LF) by Project-F, which provides the HTML5 game engine, character sprites, and physics. We modified `match.js` to broadcast game state over WebSocket and added the entire `lf2_rl/` directory for RL training.


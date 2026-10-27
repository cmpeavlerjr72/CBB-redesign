import sys, gc, tempfile
from pathlib import Path
ROOT=Path(r"C:/Users/devuser/CBB-clean-sheet"); sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import numpy as np
from cbb_sim.engine import shot_block as SBK
from cbb_sim.engine.inputs import EngineInputs
import run_engine_live as RL, build_shot_block_lut_live_v1 as SBL
inp = EngineInputs.load(ROOT/"data/processed/models/engine","F2_2025")
print("full games", len(inp.games))
sb_full = SBK.load("K2_Ocell", inp)   # what test_shot_block_engine does (cache keyed id(inp))
stale_id = id(inp); print("cached key id", stale_id, "shape", sb_full.team.shape)
# natural id reuse: free inp, then allocate slices until one lands on the freed address
del inp; gc.collect()
hit=None
for i in range(2000):
    inp2 = EngineInputs.load(ROOT/"data/processed/models/engine","F2_2025")
    gids=inp2.games["game_id"].to_numpy()[[0,7,1500,3000,4500,5600]]
    s = RL.slice_inputs(inp2,gids); s.meta["inputs_tag_loaded"]=""
    if id(s)==stale_id: hit=i; break
    if i<6: del s, inp2; gc.collect()
    if i>=6: break
print("natural reuse in", hit)
# forced equivalent: object s has a cached 5710 table under its id
s.meta.pop("shot_block_lut",None)
SBK._CACHE[("K2_Ocell", id(s))] = sb_full
SBL.attach(s, Path(tempfile.mkdtemp()))
sb = SBK.load(SBK.DEFAULT, s)
print("served shape", sb.team.shape, "len(s.games)", len(s.games))

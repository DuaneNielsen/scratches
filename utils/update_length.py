from pathlib import Path
from mentalitystorm import ObservationAction

dir_path = Path(r'c:\data\SpaceInvaders-v4\rl_raw\test')

for file in dir_path.glob('*.np'):
    oa_path = dir_path / file.with_suffix('')
    oa = ObservationAction.load(str(oa_path))
    oa.length = len(oa.reward)
    oa.save(str(oa_path))

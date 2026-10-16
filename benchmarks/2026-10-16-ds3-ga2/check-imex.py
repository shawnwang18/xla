"""Verify that only this allocation's authorized IMEX channel is visible."""
from pathlib import Path
import json
import os
import platform

channel=Path('/dev/nvidia-caps-imex-channels/channel926')
assert os.environ.get('NVIDIA_IMEX_CHANNELS') == '926', repr(os.environ.get('NVIDIA_IMEX_CHANNELS'))
assert sorted(p.name for p in channel.parent.iterdir()) == ['channel926']
fd=os.open(channel,os.O_RDWR)
info=os.fstat(fd)
os.close(fd)
print('IMEX_DEVICE',json.dumps(dict(host=platform.node(),channel=926,
      major=os.major(info.st_rdev),minor=os.minor(info.st_rdev),
      step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])),flush=True)

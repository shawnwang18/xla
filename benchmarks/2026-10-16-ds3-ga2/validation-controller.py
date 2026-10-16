"""Validate frozen binaries and all nodes, then start the authorized serial trials."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
lock=(ROOT/'validation-controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'validation-status.json').exists(), 'Reconcile previous validation first'
job=str(json.loads((ROOT/'allocation.json').read_text())['allocation_id'])
state=dict(state='WAITING_BUILDS',allocation_id=job,pid=os.getpid())

def save():
    state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path=ROOT/'validation-status.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)

save()
try:
    for attempt in range(30):
        if (ROOT/'build-controller-status.json').exists():break
        time.sleep(2)
    else:raise RuntimeError('Build controller did not publish its initial status')
    while True:
        builds=json.loads((ROOT/'build-controller-status.json').read_text())
        if builds['state']=='COMPLETED':break
        if builds['state']=='FAILED':raise RuntimeError('Build controller failed; inspect before retry')
        time.sleep(90)
    state.update(state='WAITING_IDLE');save()
    while True:
        scheduler=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'JobState=RUNNING ' in scheduler and 'NumNodes=8 ' in scheduler,scheduler
        steps=subprocess.check_output(['squeue','--steps','-j',job,'-h','-o','%i'],text=True)
        if not any(s.rsplit('.',1)[-1].isdigit() for s in steps.split()):break
        time.sleep(30)
    # Respect the per-node coordination locks until all temporary users release.
    claims=[]
    for path in sorted((ROOT/'claims').glob('*.lock')):
        claim=path.open('a')
        fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
        claims.append(claim)
    output=ROOT/'preflight';output.mkdir()
    image=json.loads((ROOT/'image.json').read_text())['image']
    image='oauth2@'+image.replace('/','#',1).replace('@sha256:',':sha256:')
    env=os.environ.copy()
    env['ENROOT_CONFIG_PATH']='/home/scratch.shawnw_inf/workload-runs/bug6811592-pr50286-dlcluster-20261014/.enroot'
    env['NVIDIA_IMEX_CHANNELS']='926'
    cmd=['srun','--jobid='+job,'--exact','--exclusive','--nodes=8','--ntasks=32',
         '--ntasks-per-node=4','--cpus-per-task=36','--gres=gpu:GB200:4',
         '--cpu-bind=cores','--gpu-bind=none','--kill-on-bad-exit=1','--time=00:15:00',
         '--job-name=ds3-preflight','--output='+str(output/'rank-%t.log'),
         '--error='+str(output/'rank-%t.log'),'--container-image='+image,
         '--container-remap-root','--container-writable','--container-env=NVIDIA_IMEX_CHANNELS',
         '--container-mounts='+','.join([str(ROOT)+':/opt/campaign',str(ROOT/'maxtext')+':/opt/maxtext',
             str(ROOT/'imex-view')+':/dev/nvidia-caps-imex-channels',
             '/dev/nvidia-caps-imex-channels/channel926:/dev/nvidia-caps-imex-channels/channel926']),
         'bash','/opt/campaign/preflight-rank.sh']
    (output/'launch.json').write_text(json.dumps(cmd,indent=2)+'\n')
    state.update(state='PREFLIGHT');save()
    with (output/'launcher.log').open('w') as log:
        rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    state['returncode']=rc
    if (output/'step.json').exists():state.update(json.loads((output/'step.json').read_text()))
    save()
    assert rc==0, 'Preflight failed; inspect rank logs'
    ranks=[json.loads(p.read_text()) for p in output.glob('rank-*.json')]
    assert len(ranks)==32 and {r['rank'] for r in ranks}==set(range(32))
    assert len({r['hostname'] for r in ranks})==8
    assert all(r['process_count']==32 and r['device_count']==32 and r['reduction']==32 for r in ranks)
    assert len({r['profile_sync_key'] for r in ranks})==1, 'Profile keys differ across GPU capacity groups'
    assert len({r['executable_cache_key'] for r in ranks})==2, 'Executable cache must distinguish the two topologies'
    assert all(r['reductions']==[32.0]*6 and r['shared_pgle_profiles'] for r in ranks)
    assert len(json.loads((output/'libraries-verified.json').read_text()))==5
    for attempt in range(12):
        accounting=subprocess.check_output(['sacct','-j',state['step_id'],'-nP','--format=JobIDRaw,State,ExitCode'],text=True)
        (output/'accounting.txt').write_text(accounting)
        if any(line.split('|')[:3]==[state['step_id'],'COMPLETED','0:0'] for line in accounting.splitlines()):break
        time.sleep(10)
    else:raise RuntimeError('Preflight accounting not confirmed; do not repeat work')
    (ROOT/'PREFLIGHT_READY').write_text(json.dumps(state)+'\n')
    state.update(state='CAMPAIGN');save()
    with (ROOT/'campaign-controller.log').open('w') as log:
        rc=subprocess.run(['python3',str(ROOT/'run-campaign.py')],stdout=log,stderr=subprocess.STDOUT).returncode
    campaign=json.loads((ROOT/'campaign-status.json').read_text())
    assert rc==0 and campaign['state']=='COMPLETED', 'Campaign did not complete; inspect exact failed step'
    state.update(state='COMPLETED');save()
except Exception as exc:
    state.update(state='FAILED',reason=str(exc));save()

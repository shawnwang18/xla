"""Run the five source trees in balanced order on one retained allocation."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import time
from collect import collect_run

ROOT = Path(__file__).resolve().parent
lock = (ROOT / 'campaign.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (ROOT / 'campaign-status.json').exists(), 'Reconcile existing campaign first'
assert json.loads((ROOT / 'build-controller-status.json').read_text())['state'] == 'COMPLETED'
assert (ROOT / 'PREFLIGHT_READY').exists(), 'Image/source/collective preflight must pass first'
job = json.loads((ROOT / 'allocation.json').read_text())['allocation_id']
state = dict(state='RUNNING', allocation_id=job, pid=os.getpid(), runs=[])


def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = ROOT / 'campaign-status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)


save()
try:
    scheduler = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    assert 'JobState=RUNNING ' in scheduler and 'NumNodes=8 ' in scheduler, scheduler
    steps = subprocess.check_output(['squeue', '--steps', '-j', job, '-h', '-o', '%i'], text=True)
    assert not any(x.rsplit('.', 1)[-1].isdigit() for x in steps.split()), steps
    image = json.loads((ROOT / 'image.json').read_text())['image']
    image = 'oauth2@' + image.replace('/', '#', 1).replace('@sha256:', ':sha256:')
    env = os.environ.copy()
    env['ENROOT_CONFIG_PATH'] = '/home/scratch.shawnw_inf/workload-runs/bug6811592-pr50286-dlcluster-20261014/.enroot'
    env['NVIDIA_IMEX_CHANNELS'] = '926'
    order = ['base', 'pr50182', 'pr50278', 'pr50281', 'pr50286',
             'pr50286', 'pr50281', 'pr50278', 'pr50182', 'base']
    variants = {v['variant']: v for v in json.loads((ROOT / 'variants.json').read_text())}
    for index, variant in enumerate(order, 1):
        output = ROOT / 'results' / f'{index:02d}-{variant}'
        output.mkdir(parents=True)
        state['runs'].append(dict(index=index, variant=variant, state='RUNNING', output=str(output)))
        save()
        cmd = ['srun', '--jobid=' + job, '--exact', '--exclusive', '--nodes=8', '--ntasks=32',
               '--ntasks-per-node=4', '--cpus-per-task=36', '--gres=gpu:GB200:4',
               '--cpu-bind=cores', '--gpu-bind=none', '--kill-on-bad-exit=1',
               '--job-name=ds3-' + variant + '-' + str(index),
               '--output=' + str(output / 'rank-%t.log'), '--error=' + str(output / 'rank-%t.log'),
               '--container-image=' + image, '--container-remap-root', '--container-writable',
               '--container-env=NVIDIA_IMEX_CHANNELS',
               '--container-mounts=' + ','.join([str(ROOT)+':/opt/campaign', str(ROOT/'maxtext')+':/opt/maxtext', str(output)+':/opt/results',
                   str(ROOT/'imex-view')+':/dev/nvidia-caps-imex-channels',
                   '/dev/nvidia-caps-imex-channels/channel926:/dev/nvidia-caps-imex-channels/channel926']),
               'bash', '/opt/campaign/run-rank.sh', variant]
        (output / 'launch.json').write_text(json.dumps(cmd, indent=2)+'\n')
        started = time.monotonic()
        with (output / 'launcher.log').open('w') as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        entry = state['runs'][-1]
        entry.update(returncode=rc, state='COMPLETED' if rc==0 else 'FAILED', wall_seconds=time.monotonic()-started)
        if (output/'step.json').exists():entry.update(json.loads((output/'step.json').read_text()))
        save()
        if rc:raise RuntimeError(f'Trial {index} {variant} failed; inspect logs before any retry')
        ranks = [json.loads(p.read_text()) for p in output.glob('rank-*.json')]
        assert len(ranks)==32 and all(r['state']=='COMPLETED' and len(r['samples'])==65 for r in ranks)
        accounting = {}
        for attempt in range(12):
            raw = subprocess.check_output(['sacct', '-j', entry['step_id'], '-nP',
                                           '--format=JobIDRaw,State,ExitCode'], text=True)
            (output / 'accounting.txt').write_text(raw)
            accounting = {f[0]: (f[1], f[2]) for line in raw.splitlines()
                          if len(f := line.split('|')) >= 3}
            if accounting.get(entry['step_id']) == ('COMPLETED', '0:0'): break
            time.sleep(10)
        validation = collect_run(entry, variants[variant]['commit'], accounting)
        (output / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        if not validation['valid']:
            entry.update(state='VALIDATION_FAILED', validation_errors=validation['errors'])
            save()
            raise RuntimeError(f'Trial {index} failed measurement validation: {validation["errors"]}')
    state.update(state='COMPLETED')
except Exception as exc:
    state.update(state='FAILED', reason=str(exc))
finally:
    save()
    # Preserve parent allocation regardless of outcome.

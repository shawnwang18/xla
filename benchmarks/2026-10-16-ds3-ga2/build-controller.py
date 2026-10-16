"""Build independent compiler variants serially inside the retained allocation."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
lock = (ROOT / 'build-controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (ROOT / 'build-controller-status.json').exists(), 'Reconcile prior controller first'
allocation = json.loads((ROOT / 'allocation.json').read_text())
job = allocation['allocation_id']
state = dict(state='WAITING_ALLOCATION', allocation_id=job, pid=os.getpid(), runs=[])


def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = ROOT / 'build-controller-status.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(p)


save()
try:
    while True:
        p = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError('Cannot reconcile allocation: ' + p.stderr)
        if 'JobState=RUNNING ' in p.stdout:
            break
        if 'JobState=PENDING ' not in p.stdout:
            raise RuntimeError('Allocation is not pending or running: ' + p.stdout)
        time.sleep(90)
    assert 'NumNodes=8 ' in p.stdout and 'UserId=shawnw(' in p.stdout, p.stdout
    (ROOT / 'allocation-running.txt').write_text(p.stdout)
    steps = subprocess.check_output(['squeue', '--steps', '-j', job, '-h', '-o', '%i|%N'], text=True)
    active = [x.split('|') for x in steps.splitlines() if x.split('|')[0].rsplit('.', 1)[-1].isdigit()]
    assert all(nodes == 'gb-nvl-053-compute09' for _, nodes in active), steps
    env = os.environ.copy()
    env['ENROOT_CONFIG_PATH'] = '/home/scratch.shawnw_inf/workload-runs/bug6811592-pr50286-dlcluster-20261014/.enroot'
    image = json.loads((ROOT / 'image.json').read_text())['image']
    image = 'oauth2@' + image.replace('/', '#', 1).replace('@sha256:', ':sha256:')
    for variant in ['base', 'pr50182', 'pr50278', 'pr50281', 'pr50286']:
        if (ROOT / 'builds' / variant / 'BUILD_READY').exists():
            state['runs'].append(dict(variant=variant, returncode=0, reused_validated_build=True))
            continue
        state.update(state='BUILDING', current_variant=variant)
        save()
        output = ROOT / 'builds' / variant
        output.mkdir(parents=True, exist_ok=True)
        cmd = ['srun', '--jobid=' + job, '--exact', '--exclusive', '--nodes=1', '--ntasks=1',
               '--nodelist=gb-nvl-053-compute01',
               '--cpus-per-task=64', '--gres=gpu:GB200:4', '--kill-on-bad-exit=1',
               '--job-name=ds3-build-' + variant, '--output=' + str(output / 'step-%j-recovery.log'),
               '--container-image=' + image, '--container-remap-root', '--container-writable',
               '--container-mounts=' + str(ROOT) + ':' + str(ROOT),
               'bash', '-c', 'task_script=$(mktemp /tmp/ds3-build.XXXXXX); cp "$1" "$task_script"; exec bash "$task_script" "$2" "$3"',
               'ds3-build', str(ROOT / ('build-clean-pr50286.sh' if variant == 'pr50286' else 'build.sh')), variant,
               '--finalize-only' if (output / 'COMPILE_READY').exists() else '']
        (output / 'launch.json').write_text(json.dumps(cmd, indent=2) + '\n')
        rc = subprocess.run(cmd, env=env).returncode
        state['runs'].append(dict(variant=variant, returncode=rc))
        save()
        if rc or not (output / 'BUILD_READY').exists():
            raise RuntimeError('Build failed for ' + variant + '; no automatic retry')
    state.update(state='COMPLETED')
except Exception as exc:
    state.update(state='FAILED', reason=str(exc))
finally:
    save()
    # The allocation holder stays alive for measurement and later reuse.

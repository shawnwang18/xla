"""Validate all ranks and report synchronized latency and distinct memory metrics."""
from pathlib import Path
import csv
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parent
VARIANTS = ['base', 'pr50182', 'pr50278', 'pr50281', 'pr50286']
GIB = 1 << 30


def collect_run(entry, expected_commit, accounting):
    output = ROOT / 'results' / Path(entry['output']).name
    ranks = [json.loads(p.read_text()) for p in sorted(output.glob('rank-*.json'))]
    expected_jax_patch = json.loads((ROOT / 'jax-pgle-profile-key.json').read_text())['files']
    errors = []
    if entry.get('returncode') != 0 or entry.get('state') != 'COMPLETED':
        errors.append('controller did not confirm successful completion')
    step = entry.get('step_id')
    if accounting.get(step) != ('COMPLETED', '0:0'):
        errors.append('exact Slurm step accounting is not COMPLETED 0:0')
    if len(ranks) != 32 or {r['rank'] for r in ranks} != set(range(32)):
        errors.append('missing or duplicate rank results')
    peaks, pools, walls = [], [], []
    step_samples = {i: [] for i in range(40, 61)}
    loss_samples = []
    for rank in ranks:
        if rank['state'] != 'COMPLETED' or rank['commit'] != expected_commit:
            errors.append(f"rank {rank['rank']} failed or has wrong source revision")
        if rank.get('jax_runtime_patch') != expected_jax_patch:
            errors.append(f"rank {rank['rank']} has unverified common JAX Python source")
        libraries = rank.get('loaded_libraries', [])
        if len(libraries) != 2 or not all(f"/builds/{entry['variant']}/jaxlibs/" in p for p in libraries):
            errors.append(f"rank {rank['rank']} did not verify loaded compiler libraries")
        samples = rank['samples']
        if len(samples) != 65 or {s['step'] for s in samples} != set(range(65)):
            errors.append(f"rank {rank['rank']} has incomplete training steps")
        for sample in samples:
            if not math.isfinite(sample['loss']) or not math.isfinite(sample['e2e_ms']) or sample['e2e_ms'] <= 0:
                errors.append('nonfinite loss or invalid latency')
            if sample['step'] in step_samples:
                step_samples[sample['step']].append(sample['e2e_ms'])
            loss_samples.append(dict(rank=rank['rank'], step=sample['step'], loss=sample['loss']))
        for device in rank['devices']:
            stats = device.get('stats') or {}
            if stats.get('peak_bytes_in_use') is not None:peaks.append(stats['peak_bytes_in_use'])
            if stats.get('peak_pool_bytes') is not None:pools.append(stats['peak_pool_bytes'])
        walls.append(rank['application_wall_seconds'])
    if len(peaks) != 32 or any(not math.isfinite(p) or p <= 0 for p in peaks):
        errors.append('missing or invalid per-GPU allocator peak counters')
    physical = []
    for path in sorted(output.glob('gpu-memory-node-*.csv')):
        per_gpu = {}
        for fields in csv.reader(path.open()):
            if not fields:continue
            try:
                _, device, used, _ = [x.strip() for x in fields]
                per_gpu.setdefault(device, []).append(float(used) / 1024)
            except (ValueError, TypeError):
                continue
        for gpu, samples in per_gpu.items():
            physical.append(dict(file=path.name, gpu=gpu, samples=len(samples), peak_gib=max(samples)))
    if len(physical)!=32 or any(p['samples']<2 for p in physical):errors.append('missing physical GPU samples')
    compiled = []
    for path in sorted(output.glob('rank-*.log')):
        text = path.read_text(errors='replace')
        values=[]
        for block in re.findall(r'Buffer Assignment Stats for jit_train_step\n(.*?)(?:\n\n)', text, re.S):
            m=re.search(r'\btotal allocation:\s+([\d.]+)(GiB|MiB|KiB|B)',block)
            if m:values.append(float(m[1])*{'GiB':1,'MiB':1/1024,'KiB':1/(1<<20),'B':1/GIB}[m[2]])
        compiled.append(dict(rank_file=path.name, allocations_gib=values))
    if len(compiled)!=32 or any(len(r['allocations_gib'])<2 for r in compiled):
        errors.append('initial and PGLE compilations not verified on all ranks')
    row=dict(variant=entry['variant'], index=entry['index'], step_id=step,
             valid=not errors, errors=errors, ranks=len(ranks), physical_devices=physical,
             compiled=compiled, losses=loss_samples)
    if not errors:
        assert all(len(v)==32 for v in step_samples.values())
        slowest=[max(step_samples[i]) for i in range(40,61)]
        row.update(e2e_step_mean_ms=statistics.mean(slowest), e2e_step_median_ms=statistics.median(slowest),
                   slowest_rank_step_ms=slowest, allocator_peak_gib=max(peaks)/GIB,
                   allocator_pool_peak_gib=max(pools)/GIB if len(pools)==32 else None,
                   sampled_physical_peak_gib=max(p['peak_gib'] for p in physical),
                   application_main_wall_seconds=max(walls),
                   initial_compiled_allocation_gib=max(r['allocations_gib'][0] for r in compiled),
                   post_pgle_compiled_allocation_gib=max(r['allocations_gib'][-1] for r in compiled))
    return row


def markdown(headers, rows):
    rows=[[str(v) for v in row] for row in rows]
    widths=[max(len(h),3,*(len(r[i]) for r in rows)) for i,h in enumerate(headers)]
    def line(row):return '| '+' | '.join(v.ljust(w) for v,w in zip(row,widths))+' |'
    return '\n'.join([line(headers),line(['-'*w for w in widths]),*(line(r) for r in rows)])


def main():
    campaign=json.loads((ROOT/'campaign-status.json').read_text())
    variants={v['variant']:v for v in json.loads((ROOT/'variants.json').read_text())}
    accounting={}
    for line in (ROOT/'accounting.txt').read_text().splitlines():
        fields=line.split('|')
        if len(fields)>=3:accounting[fields[0]]=(fields[1],fields[2])
    runs=[collect_run(e,variants[e['variant']]['commit'],accounting) for e in campaign['runs']]
    summary=[]
    for variant in VARIANTS:
        group=[r for r in runs if r['variant']==variant and r['valid']]
        if len(group)!=2:continue
        means=[r['e2e_step_mean_ms'] for r in group]
        summary.append(dict(variant=variant,runs=[r['index'] for r in group],step_ids=[r['step_id'] for r in group],
                            e2e_mean_ms=statistics.mean(means),repeat_means_ms=means,
                            repeat_stdev_ms=statistics.stdev(means),
                            peak_in_use_gib=max(r['allocator_peak_gib'] for r in group),
                            in_use_peaks_gib=[r['allocator_peak_gib'] for r in group],
                            peak_pool_gib=max(r['allocator_pool_peak_gib'] for r in group) if all(r['allocator_pool_peak_gib'] is not None for r in group) else None,
                            sampled_gpu_peak_gib=max(r['sampled_physical_peak_gib'] for r in group),
                            application_main_wall_seconds=[r['application_main_wall_seconds'] for r in group],
                            initial_compiled_gib=[r['initial_compiled_allocation_gib'] for r in group],
                            post_pgle_compiled_gib=[r['post_pgle_compiled_allocation_gib'] for r in group]))
    base=next((r for r in summary if r['variant']=='base'),None)
    if base:
        for row in summary:
            row['latency_reduction_percent']=100*(base['e2e_mean_ms']-row['e2e_mean_ms'])/base['e2e_mean_ms']
            row['speedup']=base['e2e_mean_ms']/row['e2e_mean_ms']
            row['memory_saved_gib']=base['peak_in_use_gib']-row['peak_in_use_gib']
            row['memory_reduction_percent']=100*row['memory_saved_gib']/base['peak_in_use_gib']
    result=dict(complete=campaign['state']=='COMPLETED' and len(runs)==10 and all(r['valid'] for r in runs),
                protocol='Per-step maximum synchronized E2E latency across32ranks, averaged over steps40-60 and two trials; peak memory is maximum across GPUs and both trials.',
                runs=runs,summary=summary)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    if base:
        table=markdown(['Variant','E2E (ms)','Latency reduction (%)','Speedup (x)','Peak in-use (GiB/GPU)','Saved (GiB/GPU)','Memory reduction (%)','Sampled GPU peak (GiB/GPU)','Repeat means (ms)'],
                       [["Baseline" if r['variant']=="base" else "PR #"+r['variant'][2:],f"{r['e2e_mean_ms']:.3f}",f"{r['latency_reduction_percent']:+.3f}",f"{r['speedup']:.4f}",f"{r['peak_in_use_gib']:.3f}",f"{r['memory_saved_gib']:+.3f}",f"{r['memory_reduction_percent']:+.3f}",f"{r['sampled_gpu_peak_gib']:.3f}",', '.join(f'{v:.3f}' for v in r['repeat_means_ms'])] for r in summary])
        (ROOT/'results.md').write_text(table+'\n')
        print(table)
    print(json.dumps(dict(complete=result['complete'],runs=len(runs),valid_runs=sum(r['valid'] for r in runs))))


if __name__=='__main__':main()

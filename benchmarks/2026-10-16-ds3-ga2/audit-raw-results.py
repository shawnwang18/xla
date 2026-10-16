"""Independently recompute published metrics from saved per-rank records."""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
variants = {v['variant']: v for v in json.loads((ROOT/'variants.json').read_text())}
patch = json.loads((ROOT/'jax-pgle-profile-key.json').read_text())['files']
reference_config = json.loads((ROOT/'results/01-base/rank-0.json').read_text())['resolved_config']
audit = []
for directory in sorted((ROOT/'results').iterdir()):
    validation = directory/'validation.json'
    if not validation.is_file():
        continue
    expected = json.loads(validation.read_text())
    assert expected['valid'] and not expected['errors'], directory
    records = [json.loads(p.read_text()) for p in directory.glob('rank-*.json')]
    assert len(records) == 32 and {r['rank'] for r in records} == set(range(32)), directory
    variant = expected['variant']
    timings = {}
    for record in records:
        assert record['state'] == 'COMPLETED' and record['variant'] == variant
        assert record['commit'] == variants[variant]['commit']
        assert record['tree'] == variants[variant]['tree']
        assert record['jax_runtime_patch'] == patch
        assert record['resolved_config'] == reference_config
        assert len(record['samples']) == 65
        assert {s['step'] for s in record['samples']} == set(range(65))
        assert len(record['devices']) == 1
        for sample in record['samples']:
            assert sample['rank'] == record['rank']
            assert math.isfinite(sample['loss'])
            assert math.isfinite(sample['e2e_ms']) and sample['e2e_ms'] > 0
            timings.setdefault(sample['step'], []).append(sample['e2e_ms'])
    slowest = [max(timings[step]) for step in range(40, 61)]
    e2e = statistics.mean(slowest)
    peak = max(d['stats']['peak_bytes_in_use'] for r in records for d in r['devices']) / 2**30
    physical = {}
    for path in directory.glob('gpu-memory-node-*.csv'):
        with path.open() as source:
            for fields in csv.reader(source):
                if not fields:
                    continue
                timestamp, gpu, used_mib, total_mib = [f.strip() for f in fields]
                physical.setdefault((path.name, gpu), []).append(float(used_mib)/1024)
    assert len(physical) == 32 and all(len(v) >= 2 for v in physical.values())
    physical_peak = max(max(samples) for samples in physical.values())
    assert math.isclose(e2e, expected['e2e_step_mean_ms'], abs_tol=1e-9)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(slowest, expected['slowest_rank_step_ms']))
    assert peak == expected['allocator_peak_gib']
    assert physical_peak == expected['sampled_physical_peak_gib']
    audit.append(dict(trial=directory.name, raw_records=len(records), e2e_ms=e2e,
                      peak_jax_gib=peak, sampled_physical_peak_gib=physical_peak,
                      source_and_config_verified=True, validated=True))
(ROOT/'raw-data-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(dict(validated_trials=len(audit), raw_records=sum(r['raw_records'] for r in audit))))

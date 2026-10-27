"""Local immutable run records for the provenance and mechanism studies."""
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import platform
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str)+'\n', encoding='utf-8')


class ResearchRun:
    def __init__(self, experiment, script, inputs=()):
        self.clock = time.perf_counter()
        stamp = datetime.now().astimezone()
        self.path = ROOT/'experiments/runs'/f"{experiment}_{stamp.strftime('%Y%m%dT%H%M%S%f%z')}"
        self.path.mkdir(exist_ok=False)
        config = ROOT/'experiments/configs'/f'{experiment}.json'
        self.config = json.loads(config.read_text())
        save(self.path/'config.json', self.config)
        self.manifest = dict(run_id=self.path.name, started_at=stamp.isoformat(), status='running',
            command=f'python {Path(script).relative_to(ROOT).as_posix()}', python=platform.python_version(),
            packages={k:version(k) for k in ['numpy','sympy','scipy','qiskit','qiskit-aer','seaborn','matplotlib']},
            source_sha256={}, input_sha256={}, output_sha256={}, qpu_seconds=0, qpu_jobs_submitted=0)
        sources = list((ROOT/'src/lattice_qaoa').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))
        sources += [Path(script),Path(__file__),config]
        for kind, paths in [('source', sources),('input',inputs)]:
            for p in paths:
                p = Path(p); rel = p.relative_to(ROOT)
                dest = self.path/kind/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest)
                self.manifest[kind+'_sha256'][rel.as_posix()] = sha256(p.read_bytes()).hexdigest()
        save(self.path/'manifest.json',self.manifest)
        print('RUN:',self.path,flush=True)

    def finish(self, summary=None, error=None):
        if summary is not None: save(self.path/'summary.json',summary)
        self.manifest.update(status='failed' if error else 'completed',finished_at=datetime.now().astimezone().isoformat(),
            elapsed_seconds=time.perf_counter()-self.clock)
        if error: self.manifest['error']=repr(error)
        for p in self.path.rglob('*'):
            if p.is_file() and p.name!='manifest.json' and p.relative_to(self.path).parts[0] not in ['source','input']:
                self.manifest['output_sha256'][p.relative_to(self.path).as_posix()] = sha256(p.read_bytes()).hexdigest()
        save(self.path/'manifest.json',self.manifest)
        print(self.manifest['status'].upper(),self.path,flush=True)
